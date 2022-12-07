#!/usr/bin/env python3

import gym
from garage.envs import GymEnv

from global_utils.utils import *
from airl.dmsrd import DMSRD
import csv
from garage.experiment.deterministic import set_seed


def main():
    #set_seed(0)
    demonstrations = load_expert_from_core_MSD(
        'data/LunarLander10skills500.pkl', length=500,
        repeat_each_skill=3,
        separate_styles=True)
    env = GymEnv('LunarLanderContinuous-v2', max_episode_length=500)
    env_test = gym.make('LunarLanderContinuous-v2')

    dmsrd = DMSRD(env, env_test, demonstrations, trajs_path='lunar_trajs', reward_path='LunarTestReward',
                  bool_save_vid=False, log_prefix='lunar_lander_dmsrd',
                  n_workers_irl=10, batch_when_finding_mixture=9, grid_shots=900, mix_repeats=3,
                  episode_length=500, airl_itrs=1000, msrd_itrs=20)

    with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Order of Demos"])
        csvwriter.writerow(dmsrd.rand_idx)

    for iteration in range(len(demonstrations)):
        dmsrd.new_demonstration(iteration)
        dmsrd.mixture_optimize(iteration)
        dmsrd.msrd(iteration)

    with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows([["Order of Demos"]])
        csvwriter.writerow(dmsrd.rand_idx)
        csvwriter.writerows([["Mixtures"]])
        csvwriter.writerows(dmsrd.mixture_weights)


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
