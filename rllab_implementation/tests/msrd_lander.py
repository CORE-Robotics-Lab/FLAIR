import os

import numpy as np
from datetime import datetime
import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.msd import ReLUModel, AIRLMultiStyleSingle
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
from inverse_rl.models.dmsrd import new_likelihood
import rllab.misc.logger as logger
# from tests.heatmap import heatmap, annotate_heatmap
# import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_tree(x, xp, k):
    # https://github.com/BiuBiuBiLL/NPEET_LNC/blob/master/lnc.py
    # https://github.com/scipy/scipy/issues/9890 p=2 or np.inf
    tree = cKDTree(x)
    return np.clip(tree.query(xp, k=k + 1, p=float('inf'))[0][:, k], 1e-30, None) # chebyshev distance of k+1-th nearest neighbor


def main():
    env = TfEnv(CustomGymEnv('LunarLanderContinuous-v2', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/LunarLander10skills500.pkl', length=500, repeat_each_skill=3,
                                                     separate_styles=True)


    # Hyperparameters
    state_only = True
    num_skills = len(demonstrations)
    n_epochs = 2000
    repeat_each_epoch = 1
    discriminator_update_step = 10
    now = datetime.now()
    log_path = f"data/lunar_lander_msrd/{now.strftime('%d_%m_%Y_%H_%M_%S')}"
    assert not os.path.exists(log_path), "log path already exist! "

    save_dictionary = {}

    center_reward = ReLUModel("center", env.observation_space.shape[0]) \
        if state_only \
        else ReLUModel("center", env.observation_space.shape[0] + env.action_space.shape[0])

    for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='center')):
        save_dictionary[f'my_center_{idx}'] = var

    n_timesteps = 500
    deterministic = False

    reward_fs = []
    policies = []
    algos = []
    for skill in range(num_skills):
        irl_model = AIRLMultiStyleSingle(env, center_reward,
                                         expert_trajs=demonstrations[skill],
                                         state_only=state_only, fusion=True, max_itrs=discriminator_update_step, name=f'skill_{skill}')

        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{skill}')):
            save_dictionary[f'my_skill_{skill}_{idx}'] = var
        policy = GaussianMLPPolicy(name=f'policy_{skill}', env_spec=env.spec, hidden_sizes=(32, 32))
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{skill}')):
            save_dictionary[f'my_policy_{skill}_{idx}'] = var
        algo = IRLTRPO(
            env=env,
            policy=policy,
            irl_model=irl_model,
            n_itr=1500,  # doesn't matter, will change
            batch_size=4000,
            max_path_length=500,
            discount=0.99,
            store_paths=False,
            pol_updates=2,
            discrim_train_itrs=discriminator_update_step,
            irl_model_wt=1.0,
            entropy_weight=0,  # from AIRL: this should be 1.0 but 0.1 seems to work better
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            gae_lambda=0.98,
            step_size=0.01,
            optimizer_args=dict(cg_iters=15)
        )
        policies.append(policy)
        reward_fs.append(irl_model)
        algos.append(algo)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    assert not os.path.exists(log_path), "log path already exist! "
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            center_reward_gradients = None
            for skill in range(num_skills):
                with rllab_logdir(algo=algos[skill], dirname=log_path+'/skill_%d' % skill):
                    algos[skill].start_itr = repeat_each_epoch*epoch
                    algos[skill].n_itr = repeat_each_epoch*(epoch+1)
                    algos[skill].train()
                    if center_reward_gradients is None:
                        center_reward_gradients = algos[skill].center_reward_gradients
                    else:
                        assert center_reward_gradients.keys() == algos[skill].center_reward_gradients.keys()
                        for key in center_reward_gradients.keys():
                            center_reward_gradients[key] += algos[skill].center_reward_gradients[key]
            feed_dict = {}
            assert center_reward.grad_map_vars.keys() == center_reward_gradients.keys()
            for key in center_reward.grad_map_vars.keys():
                feed_dict[center_reward.grad_map_vars[key]] = center_reward_gradients[key]
            sess.run(center_reward.step, feed_dict=feed_dict)

        post_probs = [new_likelihood(np.array(reward_fs[0].eval_expert_probs(demonstrations[i], policies[i])))
                      for i in range(len(demonstrations))]

        with rllab_logdir(algo=algos[0], dirname=log_path + '/likelihood'):
            for demo_ind in range(len(demonstrations)):
                post_rew = get_reward(env, policies[demo_ind])
                divergence = get_divergence(env, policies[demo_ind], demonstrations[demo_ind])
                logger.record_tabular(f"Demonstration {demo_ind}", post_rew)
                logger.record_tabular(f"Divergence {demo_ind}", divergence)
            d = [i for i in range(len(demonstrations))]
            logger.record_tabular(f'Demonstrations', d)
            logger.record_tabular(f'New_Likelihoods', post_probs)
            logger.dump_tabular(with_prefix=False, write_header=True)

        for skill in range(num_skills):
            # Rollout a trajectory
            ob = env.reset()
            # env.render()
            done = False
            # obs, acts = [], []
            imgs = []
            idx = 0
            policy = policies[skill]
            for timestep in range(n_timesteps):
                act = policy.get_action(ob)
                # print(act)
                # obs.append(ob)
                act_executed = act[1]["mean"] if deterministic else act[0]
                # acts.append(act_executed)
                ob, rew, done, info = env.step(act_executed)
                imgs.append(env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{skill}.avi"))

        saver = tf.train.Saver(save_dictionary)
        saver.save(sess, f"{log_path}/model.ckpt")


def get_reward(env, policy):
    episode_return = 0.0
    for _ in range(10):
        ob = env.reset()
        for timestep in range(1000):
            act = policy.get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            episode_return += rew
    return episode_return/10


def get_divergence(env, policy, demonstration):
    dist = 0.0
    for _ in range(10):
        obs = []
        ob = env.reset()
        for timestep in range(1000):
            act = policy.get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            obs.append(ob)

        expert = demonstration[0]["observations"]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 3)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    return dist/10


if __name__ == "__main__":
    main()
