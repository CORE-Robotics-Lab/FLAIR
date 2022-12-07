#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
import os
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer
from global_utils.utils import *
from garage.experiment import Snapshotter

# snapshotter = Snapshotter()
# with TFTrainer(snapshotter) as trainer:
#     env = GymEnv('InvertedDoublePendulum-v2')
#
#     policy = GaussianMLPPolicy(name='policy',
#                                   env_spec=env.spec,
#                                   hidden_sizes=(32, 32))
#
#     baseline = LinearFeatureBaseline(env_spec=env.spec)
#
#     sampler = LocalSampler(agents=policy,
#                            envs=env,
#                            max_episode_length=env.spec.max_episode_length,
#                            is_tf_worker=True)
#
#     algo = TRPO(env_spec=env.spec,
#                 policy=policy,
#                 baseline=baseline,
#                 sampler=sampler,
#                 discount=0.99,
#                 max_kl_step=0.01)
#
#     trainer.setup(algo, env)
#     trainer.train(n_epochs=100, batch_size=4000)
#
#     env = gym.make('InvertedDoublePendulum-v2')
#     for repeat in range(3):
#         ob = env.reset()
#         policy.reset()
#         imgs = []
#         for timestep in range(1000):
#             ob, rew, done, info = env.step(policy.get_action(ob)[0])
#             imgs.append(env.render('rgb_array'))
#         save_video(imgs, os.path.join(f"policy_videos/skill_{repeat}.avi"))
#     env.close()

snapshotter = Snapshotter()
with TFTrainer(snapshotter) as trainer:
    trainer.restore('data/local/experiment/')
    trainer.resume(n_epochs=500, batch_size=4000)
    env = gym.make('InvertedDoublePendulum-v2')
    for repeat in range(3):
        ob = env.reset()
        policy = trainer._algo.policy
        policy.reset()
        imgs = []
        for timestep in range(1000):
            ob, rew, done, info = env.step(policy.get_action(ob)[0])
            imgs.append(env.render('rgb_array'))
        save_video(imgs, os.path.join(f"policy_videos/skill_{repeat}.avi"))
    env.close()