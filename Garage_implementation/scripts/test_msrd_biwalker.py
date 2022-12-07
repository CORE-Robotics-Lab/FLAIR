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
from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from airl.irl_trpo import TRPO
from models.msd import ReLUModel, AIRLMultiStyleSingle

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
from airl.test_performance import *
import matplotlib
import matplotlib.pyplot as plt
import csv

now = datetime.now()
log_path = f"data/bipedal_walker_msrd/21_01_2022_11_54_13"

irl_models = []
policies = []
algos = []
trainers = []

demonstrations = load_expert_from_core_MSD(
    'data/BipedalWalker10skills.pkl', length=1000,
    repeat_each_skill=3,
    separate_styles=True)
env = GymEnv('BipedalWalker-v2')

n_epochs = 2500
timesteps = 1000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}

    center_reward = ReLUModel("center", env.observation_space.shape[0])

    for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='center')):
        save_dictionary[f'my_center_{idx}'] = var

    for index in range(len(demonstrations)):
        irl_model = AIRLMultiStyleSingle(env, center_reward,
                                         expert_trajs=demonstrations[index],
                                         state_only=True, fusion=True, max_itrs=10, name=f'skill_{index}')
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

    saver = tf.train.Saver(save_dictionary)
    saver.restore(sess, f"{log_path}/model.ckpt")

    env = gym.make('BipedalWalker-v2')

    # for i in range(len(policies)):
    #     ob = env.reset()
    #     policy = policies[i]
    #     imgs = []
    #     for timestep in range(500):
    #         ob, rew, done, info = env.step(policy.get_action(ob)[0])
    #         imgs.append(env.render('rgb_array'))
    #     save_video(imgs, os.path.join(f"{log_path}/policy_videos_test/skill_{i}.avi"))

    get_divergence(env, policies, demonstrations, timesteps)

    # with open(f'{log_path}/likelihood.csv', 'w') as csvfile:
    #     # creating a csv writer object
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["Likelihoods"])
    #     csvwriter.writerow(
    #         get_likelihoods(irl_model, demonstrations, policies))
        # csvwriter.writerow(["Rewards"])
        # csvwriter.writerow(get_reward(env, policies, timesteps))
        # csvwriter.writerow(["Divergences"])
        # csvwriter.writerow(
        #     get_divergence(env, policies, demonstrations, timesteps))

    # trajectories = []
    # for i in range(40):
    #     with open(f'data/lunar_trajs/trajectories_{i}.pkl', "rb") as f:
    #         trajectories.extend(pickle.load(f))
    #
    # record = np.zeros(len(trajectories))
    # for tidx, traj in enumerate(trajectories):
    #     reward_cent = tf.get_default_session().run(
    #         irl_model.reward_center,
    #         feed_dict={irl_model.obs_t: traj["observations"]})
    #     score = reward_cent[:, 0]
    #     record[tidx] = np.mean(score)

    # with open(f'{log_path}/reward.csv', 'w') as csvfile:
    #     # creating a csv writer object
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["Task Reward"])
    #     csvwriter.writerow(record.tolist())

    # rew = []
    # for demo in demonstrations:
    #     strat_rew = []
    #     for strat in range(len(irl_models)):
    #         rew_repeat = 0
    #         for traj in demo:
    #             reward = tf.get_default_session().run(
    #                 irl_models[strat].reward_peri,
    #                 feed_dict={
    #                     irl_models[strat].obs_t: traj["observations"]})
    #             score = np.mean(reward[:, 0])
    #             rew_repeat += np.mean(score)
    #         strat_rew.append(rew_repeat/len(demo))
    #     rew.append(strat_rew)
    #
    # rew = np.array(rew)
    #
    # for j in range(len(rew[0])):
    #     max_reward = np.amax(rew[:, j])
    #     rew[:, j] = np.exp(rew[:, j] - max_reward)
    #
    # name = [f'Demonstration {i}' for i in range(len(rew))]
    # trajectories = [f'Strategy {i}' for i in range(len(rew[0]))]
    #
    # fig, ax = plt.subplots()
    #
    # im, cbar = heatmap(rew, name, trajectories, ax=ax,
    #                    cmap="YlGn", cbarlabel="reward")
    # texts = annotate_heatmap(im)
    #
    # fig.tight_layout()
    # plt.savefig(f'{log_path}/heatmap.png')
    # plt.close()
