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
# from airl.irl_trpo import TRPO
# from models.airl_state import AIRL

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer
from global_utils.utils import *
from garage.experiment import Snapshotter

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


log_path = f"data/lunar_lander_airl_single/28_10_2021_13_08_56"

irl_models = []
policies = []
algos = []

demonstrations = load_expert_from_core_MSD(
    'data/LunarLander10skills500.pkl', length=1000,
    repeat_each_skill=3,
    separate_styles=True)
demonstrations = [demonstrations[0]]
env = GymEnv('LunarLanderContinuous-v2')

def get_rewdiv(env, policy, demonstration):
    episode_return = 0.0
    dist = 0.0
    for _ in range(10):
        obs = []
        ob = env.reset()
        policy.reset()
        for timestep in range(1000):
            act = policy.get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            obs.append(ob)
            episode_return += rew

        expert = demonstration[0]["observations"]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 3)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    return episode_return/10, dist/10


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    snapshotter = Snapshotter(log_path)
    with TFTrainer(snapshotter, sess=sess) as trainer:
        save_dictionary = {}
        for index in range(len(demonstrations)):
            # irl_model = AIRL(env=env, expert_trajs=demonstrations[index],
            #                  state_only=True, fusion=False,
            #                  max_itrs=10,
            #                  name=f'skill_{index}')
            # for idx, var in enumerate(
            #     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            #                       scope=f'skill_{index}')):
            #     save_dictionary[f'my_skill_{index}_{idx}'] = var

            policy = GaussianMLPPolicy(name=f'policy_{index}',
                                       env_spec=env.spec,
                                       hidden_sizes=(32, 32))
            for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'policy_{index}')):
                save_dictionary[f'my_policy_{index}_{idx}'] = var

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(agents=policy,
                                   envs=env,
                                   max_episode_length=env.spec.max_episode_length,
                                   is_tf_worker=True)

            # algo = TRPO(env_spec=env.spec,
            #             policy=policy,
            #             baseline=baseline,
            #             index=index,
            #             sampler=sampler,
            #             irl_model=irl_model,
            #             # name=f'TRPO_{index}',
            #             # scope=f'NNO_{index}',
            #             discount=0.99,
            #             max_kl_step=0.01)
            # irl_models.append(irl_model)
            policies.append(policy)
            # algos.append(algo)

        # for i in range(len(demonstrations)):
        #     trainer.setup(algos[i], env)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model.ckpt")

        env = gym.make('LunarLanderContinuous-v2')
        # print([get_rewdiv(env, policies[i], demonstrations[i]) for i in range(len(demonstrations))])

        for i in range(len(demonstrations)):
            # trainer.train(n_epochs=500, batch_size=2000)
            ob = env.reset()
            episode_reward = 0.0
            policy = policies[i]
            policy.reset()
            imgs = []
            for timestep in range(1000):
                ob, rew, done, info = env.step(policy.get_action(ob)[0])
                episode_reward += rew
                imgs.append(env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{i}.avi"))
            print(episode_reward)
        env.close()

        saver = tf.train.Saver(save_dictionary)
        saver.save(sess, f"{log_path}/model.ckpt")

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

