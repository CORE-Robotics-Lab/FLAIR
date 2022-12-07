import pickle
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger
import matplotlib
import matplotlib.pyplot as plt
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
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstration data from this repo's data generation
    # experts = load_latest_experts('data/inverted_pendulum', n=5, min_return=-10)

    # load expert demonstration data from core_MSD's data generation
    # experts = load_expert_from_core_MSD("/home/zac/Programming/core_MSD/dataset/old_datasets/InvertedPendulum-v2-nonranked-multistyle.pkl",
    #                                     repeat_each_skill=3,
    #                                     separate_styles=False)

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000,  repeat_each_skill=3,
                                                     separate_styles=True)

    # load mixture
    # demonstrations = load_expert_from_core_MSD("data/038_0.6_0.1_0.3_"+"expertmix.pkl", length=1000, repeat_each_skill=3,
    #                                                  separate_styles=True)

    # Hyperparameters
    discriminator_update_step = 10
    n_epochs = 2140
    log_path = "data/inverted_pendulum_airl_single/30_10_2021_19_12_51"

    n_timesteps = 1000
    deterministic = False

    irl_models = []
    policies = []
    algos = []

    save_dictionary = {}
    for index in range(len(demonstrations)):
        irl_model = AIRL(env=env, expert_trajs=demonstrations[index], state_only=True, fusion=True,
                         max_itrs=discriminator_update_step, name=f'skill_{index}')
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{index}')):
            save_dictionary[f'my_skill_{index}_{idx}'] = var

        policy = GaussianMLPPolicy(name=f'policy_{index}', env_spec=env.spec, hidden_sizes=(32, 32))
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{index}')):
            save_dictionary[f'my_policy_{index}_{idx}'] = var

        algo = IRLTRPO(
            env=env,
            policy=policy,
            irl_model=irl_model,
            n_itr=n_epochs,
            batch_size=10000,
            max_path_length=1000,
            discount=0.99,
            store_paths=False,
            center_grads=False,
            discrim_train_itrs=discriminator_update_step,
            irl_model_wt=1.0,
            entropy_weight=0.1,  # from AIRL: this should be 1.0 but 0.1 seems to work better
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=env.spec)
        )
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    trajectories = []
    for i in range(20):
        with open(f'data/trajs/trajectories_{i}.pkl', "rb") as f:
            trajectories.extend(pickle.load(f))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for idx in range(len(algos)):
            start = time.time()
            margin = 0
            rewards = []
            divergences = []
            rewards_time = []
            divergences_time = []
            with rllab_logdir(algo=algos[idx], dirname=log_path+'/demo_%d' % idx):
                for epoch in range(n_epochs):
                    algos[idx].start_itr = epoch
                    algos[idx].n_itr = (epoch+1)
                    algos[idx].train()
                    if epoch % 20 == 0:
                        rew, div = get_rewdiv(env, policies[idx], demonstrations[idx])
                        rewards.append(rew)
                        divergences.append(div)
                    if time.time() - start > margin:
                        rew, div = get_rewdiv(env, policies[idx], demonstrations[idx])
                        rewards_time.append(rew)
                        divergences_time.append(div)
                        margin += 500

            with rllab_logdir(algo=algo, dirname=log_path + f'/sample_{idx}'):
                logger.record_tabular(f'Rewards', rewards)
                logger.record_tabular(f'Divergences', divergences)
                logger.record_tabular(f'Rewards_Time', rewards_time)
                logger.record_tabular(f'Divergences_Time', divergences_time)
                logger.dump_tabular(with_prefix=False, write_header=True)

        saver = tf.train.Saver(save_dictionary)
        saver.save(sess, f"{log_path}/model.ckpt")


def get_rewdiv(env, policy, demonstration):
    dist = 0.0
    episode_return = 0.0
    for _ in range(10):
        obs = []
        ob = env.reset()
        for timestep in range(1000):
            act = policy.get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            episode_return += rew
            obs.append(ob)

        expert = demonstration[0]["observations"]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 3)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    return episode_return/10, dist/10


if __name__ == "__main__":
    main()
