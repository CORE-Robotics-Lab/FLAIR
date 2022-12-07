import os
import pickle
from datetime import datetime
import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
import numpy as np

from inverse_rl.algos.trpo import TRPO
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.likelihood_utils import new_likelihood
from global_utils.utils import *
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
    return tree.query(xp, k=k + 1, p=float('inf'))[0][:, k] # chebyshev distance of k+1-th nearest neighbor


def main():
    # env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))
    env = TfEnv(CustomGymEnv('InvertedDoublePendulum-v2', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    # demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000,  repeat_each_skill=3,
    #                                                  separate_styles=True)
    demonstrations = load_expert_from_core_MSD('data/InvertedDoublePendulum10skills.pkl', length=1000, repeat_each_skill=3,
                                                     separate_styles=True)

    # Hyperparameters
    discriminator_update_step = 10
    n_epochs = 3000

    n_timesteps = 1000
    deterministic = False

    now = datetime.now()
    log_path = f"data/double_pendulum_airl_single/{now.strftime('%d_%m_%Y_%H_%M_%S')}"
    assert not os.path.exists(log_path), "log path already exist! "
    # log_path = "data/inverted_pendulum_airl_single/20_10_2021_13_20_10"

    irl_models = []
    policies = []
    algos = []

    save_dictionary = {}
    for index in range(len(demonstrations)):
        irl_model = AIRL(env=env, expert_trajs=demonstrations[index], state_only=True, fusion=False,
                         max_itrs=discriminator_update_step, name=f'skill_{index}')
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{index}')):
            save_dictionary[f'my_skill_{index}_{idx}'] = var

        policy = GaussianMLPPolicy(name=f'policy_{index}', env_spec=env.spec,
                                   hidden_sizes=(32, 32))
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{index}')):
            save_dictionary[f'my_policy_{index}_{idx}'] = var

        algo = IRLTRPO(
            env=env,
            policy=policy,
            irl_model=irl_model,
            n_itr=n_epochs,
            batch_size=10000,
            max_path_length=n_timesteps,
            discount=0.99,
            store_paths=False,
            discrim_train_itrs=discriminator_update_step,
            irl_model_wt=1.0,
            pol_updates=2,
            entropy_weight=0,  # from AIRL: this should be 1.0 but 0.1 seems to work better
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=env.spec),
            gae_lambda=0.97,
            step_size=0.01,
            optimizer_args=dict(cg_iters=15)
        )
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    # trajectories = []
    # for i in range(20):
    #     with open(f'data/trajs/trajectories_{i}.pkl', "rb") as f:
    #         trajectories.extend(pickle.load(f))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for idx in range(len(algos)):
            with rllab_logdir(algo=algos[idx], dirname=log_path+'/demo_%d' % idx):
                algos[idx].train()
                # Rollout a trajectory
                ob = env.reset()
                imgs = []
                policy = policies[idx]
                for timestep in range(n_timesteps):
                    act = policy.get_action(ob)
                    act_executed = act[1]["mean"] if deterministic else act[0]
                    ob, rew, done, info = env.step(act_executed)
                    imgs.append(env.render('rgb_array'))
                save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{idx}.avi"))

        # with rllab_logdir(algo=algo, dirname=log_path + '/reward'):
        #     for traj in trajectories:
        #         for demo in range(len(demonstrations)):
        #             reward = tf.get_default_session().run(irl_models[demo].reward,
        #                                                   feed_dict={irl_models[demo].obs_t: traj})
        #             score = reward[:, 0]
        #             logger.record_tabular(f'Demonstration_{demo}', np.mean(score))
        #         logger.dump_tabular(with_prefix=False, write_header=False)

        post_probs = [new_likelihood(np.array(irl_models[0].eval_expert_probs(demonstrations[i], policies[i])))
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
