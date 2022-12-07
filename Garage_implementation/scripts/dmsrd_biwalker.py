#!/usr/bin/env python3

import gym
from garage.envs import GymEnv

from global_utils.utils import *
from airl.dmsrd import DMSRD
import csv


def main():
    demonstrations = load_expert_from_core_MSD(
        'data/BipedalWalker10skills.pkl', length=1000,
        repeat_each_skill=3,
        separate_styles=True)[:10]
    env = GymEnv('BipedalWalker-v3', max_episode_length=1000)
    env_test = gym.make('BipedalWalker-v3')

    dmsrd = DMSRD(env, env_test, demonstrations, trajs_path='biwalker_trajs', reward_path='BipedalTestReward', n_workers_irl=10, batch_when_finding_mixture=3, grid_shots=500, bool_save_vid=False, log_prefix='biwalker_dmsrd', episode_length=1000, airl_itrs=1800, msrd_itrs=10, start_msrd=3, mix_repeats=3, new_strategy_threshold=8)
    # dmsrd.rand_idx = np.arange(10)
    np.random.shuffle(dmsrd.rand_idx)

    with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Demonstrations"])
        csvwriter.writerow(dmsrd.rand_idx)

    for iteration in range(len(demonstrations)):
        dmsrd.new_demonstration(iteration)
        dmsrd.mixture_optimize(iteration)
        dmsrd.msrd(iteration)

    with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Mixtures"])
        csvwriter.writerows(dmsrd.mixture_weights)


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
