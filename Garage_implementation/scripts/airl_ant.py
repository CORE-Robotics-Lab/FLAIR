#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
import os
import csv
from datetime import datetime
import gym
import tensorflow as tf
from garage import wrap_experiment
import dowel
from dowel import logger
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from airl.irl_trpo import TRPO
from models.airl_state import AIRL

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
from airl.test_performance import *

now = datetime.now()
log_path = f"data/ant_pendulum_airl_single/{now.strftime('%d_%m_%Y_%H_%M_%S')}"

irl_models = []
policies = []
algos = []
trainers = []

demonstrations = load_expert_from_core_MSD(
    'data/Ant10skills.pkl', length=1000,
    repeat_each_skill=3,
    separate_styles=True)
env = GymEnv('Ant-v3')
timesteps = 1000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}
    for index in range(len(demonstrations)):
        snapshotter = Snapshotter(f'{log_path}/skill_{index}')
        trainer = Trainer(snapshotter)

        irl_model = AIRL(env=env, expert_trajs=demonstrations[index],
                         state_only=True, fusion=True,
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

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    index=index,
                    sampler=sampler,
                    irl_model=irl_model,
                    generator_train_itrs=1,
                    gae_lambda=0.97,
                    policy_ent_coeff=0.001,
                    entropy_method='regularized',
                    # center_adv=False,
                    # stop_entropy_gradient=True,
                    discount=0.99,
                    max_kl_step=0.01)
        trainers.append(trainer)
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    sess.run(tf.global_variables_initializer())
    env_test = gym.make('Ant-v3')
    for i in range(len(demonstrations)):
        trainer = trainers[i]

        sampler = RaySampler(agents=policies[i],
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True)

        logger.remove_all()
        tabular_log_file = os.path.join(trainer._snapshotter.snapshot_dir, 'progress.csv')
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(dowel.StdOutput())
        algos[i]._sampler = sampler
        trainer.setup(algos[i], env)
        trainer.train(n_epochs=2000, batch_size=40000)
        ob = env_test.reset()
        policy = policies[i]
        imgs = []
        for timestep in range(timesteps):
            ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
            imgs.append(env_test.render('rgb_array'))
        save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{i}.avi"))
    env_test.close()

    saver = tf.train.Saver(save_dictionary)
    saver.save(sess, f"{log_path}/model.ckpt")

    with open(f'{log_path}/likelihood.csv', 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Likelihoods"])
        csvwriter.writerow(get_likelihoods(irl_model, demonstrations, policies))
        csvwriter.writerow(["Rewards"])
        csvwriter.writerow(get_reward(env_test, policies, timesteps))
        csvwriter.writerow(["Divergences"])
        csvwriter.writerow(get_divergence(env_test, policies, demonstrations, timesteps))

