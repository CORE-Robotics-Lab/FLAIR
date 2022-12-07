#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
from datetime import datetime
import gym
import tensorflow as tf
import dowel
from dowel import logger
from garage.envs import GymEnv
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
log_path = f"data/lunar_lander_airl_efficiency/{now.strftime('%d_%m_%Y_%H_%M_%S')}"

irl_models = []
policies = []
algos = []
trainers = []

demonstrations = load_expert_from_core_MSD(
    'data/LunarLander10skills500.pkl', length=500,
    repeat_each_skill=3,
    separate_styles=True)
env = GymEnv('LunarLanderContinuous-v2')

timesteps = 500
n_epochs = 2000

dmsrd_truths = [-6539.080433, -6939.855983, -6266.952583, -13761.25092,
                -14766.96227, -12899.21675, -22856.65367, -18065.55753,
                -20633.05733, -22779.72083]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}
    for index in range(len(demonstrations)):
        snapshotter = Snapshotter(f'{log_path}/skill_{index}')
        trainer = Trainer(snapshotter)

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

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = None

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    index=index,
                    sampler=sampler,
                    irl_model=irl_model,
                    generator_train_itrs=2,
                    discrim_train_itrs=10,
                    discount=0.99,
                    max_kl_step=0.01)
        trainers.append(trainer)
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    sess.run(tf.global_variables_initializer())
    env_test = gym.make('LunarLanderContinuous-v2')

    with open(f'{log_path}/efficiency.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Likelihoods"])
    with open(f'{log_path}/efficiency_epochs.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Epochs"])

    for i in range(len(demonstrations)):
        trainer = trainers[i]
        trainer.setup(algos[i], env)
        logger.remove_all()
        tabular_log_file = os.path.join(trainer._snapshotter.snapshot_dir, 'progress.csv')
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(dowel.StdOutput())
        sampler = RaySampler(agents=policies[i],
                               envs=env,
                               max_episode_length=timesteps,
                               is_tf_worker=True,
                             n_workers=10)
        trainer._sampler = sampler
        trainer._start_worker()

        likelihoods_epoch = []
        threshold_epoch = [0]
        first_epoch = True
        for epoch in range(n_epochs):
            trainer.train(n_epochs=epoch + 1, batch_size=10000, start_epoch=epoch)
            likelihood = np.sum(get_likelihoods(irl_model, [demonstrations[i]], [policies[i]]))
            likelihoods_epoch.append(likelihood)
            if epoch > 30 and first_epoch and likelihood > dmsrd_truths[i]:
                threshold_epoch.append(epoch)
                first_epoch = False

        with open(f'{log_path}/efficiency.csv', 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(likelihoods_epoch)
        with open(f'{log_path}/efficiency_epochs.csv', 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(threshold_epoch)

        trainer._shutdown_worker()
        ob = env_test.reset()
        # policy = policies[i]
        # imgs = []
        # for timestep in range(timesteps):
        #     ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
        #     imgs.append(env_test.render('rgb_array'))
        # save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{i}.avi"))

    with open(f'{log_path}/likelihood.csv', 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        likelihoods = get_likelihoods(irl_model, demonstrations, policies)
        rewards = get_reward(env_test, policies, timesteps)
        divergences = get_divergence(env_test, policies, demonstrations, timesteps)
        csvwriter.writerow(["Likelihoods"])
        csvwriter.writerow(likelihoods)
        csvwriter.writerow(["Rewards"])
        csvwriter.writerow(rewards)
        csvwriter.writerow(["Divergences"])
        csvwriter.writerow(divergences)
        csvwriter.writerow(["Mean_Likelihoods"])
        csvwriter.writerow([np.mean(likelihoods)])
        csvwriter.writerow(["Mean_Rewards"])
        csvwriter.writerow([np.mean(rewards)])
        csvwriter.writerow(["Mean_Divergences"])
        csvwriter.writerow([np.mean(divergences)])

    saver = tf.train.Saver(save_dictionary)
    saver.save(sess, f"{log_path}/model.ckpt")
