#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
import os
from datetime import datetime
import gym
import tensorflow as tf
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from airl.irl_trpo import TRPO
from models.airl_state import AIRL

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer
from global_utils.utils import *
from garage.experiment import Snapshotter
from airl.test_performance import *

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


log_path = f"data/lunar_lander_airl_single/17_01_2022_19_23_08"

irl_models = []
policies = []
algos = []

timesteps = 500

demonstrations = load_expert_from_core_MSD(
    'data/LunarLander10skills500.pkl', length=500,
    repeat_each_skill=3,
    separate_styles=True)
env = GymEnv('LunarLanderContinuous-v2')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}
    for index in range(len(demonstrations)):
        irl_model = AIRL(env=env, expert_trajs=demonstrations[index],
                         state_only=True, fusion=False,
                         max_itrs=10,
                         name=f'skill_{index}')
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'skill_{index}')):
            save_dictionary[f'my_skill_{index}_{idx}'] = var

        policy = GaussianMLPPolicy(name=f'policy_{index}',
                                   env_spec=env.spec,
                                   hidden_sizes=(32, 32))
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'policy_{index}')):
            save_dictionary[f'my_policy_{index}_{idx}'] = var

        irl_models.append(irl_model)
        policies.append(policy)

    # sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(save_dictionary)
    saver.restore(sess, f"{log_path}/model.ckpt")

    env = gym.make('LunarLanderContinuous-v2')
    # for i in range(len(policies)):
    #     ob = env.reset()
    #     policy = policies[i]
    #     imgs = []
    #     for timestep in range(timesteps):
    #         ob, rew, done, info = env.step(policy.get_action(ob)[0])
    #         imgs.append(env.render('rgb_array'))
    #     save_video(imgs, os.path.join(f"{log_path}/policy_videos_test/skill_{i}.avi"))

    get_divergence(env, policies, demonstrations, timesteps)

    # with open(f'{log_path}/probs.csv', 'w') as csvfile:
        # creating a csv writer object
        # csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(["Likelihoods"])
        # csvwriter.writerow(
        #     get_likelihoods(irl_model, demonstrations, policies))
    #     csvwriter.writerow(["Rewards"])
    #     csvwriter.writerow(get_reward(env, policies, timesteps))
    #     csvwriter.writerow(["Divergences"])
    #     csvwriter.writerow(
    #         get_divergence(env, policies, demonstrations, timesteps))
    #
    # env.close()

# snapshotter = Snapshotter()
# with TFTrainer(snapshotter) as trainer:
#     trainer.restore('data/local/experiment/')
#     trainer.resume(n_epochs=500, batch_size=4000)
#     env = gym.make('InvertedDoublePendulum-v2')
#     for repeat in range(3):
#         ob = env.reset()
#         policy = trainer._algo.policy
#         policy.reset()
#         imgs = []
#         for timestep in range(1000):
#             ob, rew, done, info = env.step(policy.get_action(ob)[0])
#             imgs.append(env.render('rgb_array'))
#         save_video(imgs, os.path.join(f"policy_videos/skill_{repeat}.avi"))
#     env.close()

