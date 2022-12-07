#!/usr/bin/env python3

import gym
import numpy as np

from garage.envs import GymEnv

from global_utils.utils import *
from airl.dmsrd_scale_new import DMSRDScale
import csv
from garage.experiment.deterministic import set_seed


def main():
    set_seed(0)
    demonstrations = load_expert_from_core_MSD(
        'data/LunarLander100mix.pkl', length=500,
        repeat_each_skill=3,
        separate_styles=True, squeeze_option=3)
    demonstrations = demonstrations + demonstrations
    env = GymEnv('LunarLanderContinuous-v2', max_episode_length=500)
    env_test = gym.make('LunarLanderContinuous-v2')

    rewards, divergences, likelihoods, ground_truths, ablations = [], [], [], [], []

    dmsrd = DMSRDScale(env, env_test, demonstrations, trajs_path='lunar_trajs', reward_path='LunarTestReward',
                  bool_save_vid=False, log_prefix='lunar_lander_dmsrd_scale', new_strategy_threshold=10.0,
                  n_workers_irl=10, batch_when_finding_mixture=9, grid_shots=500, mix_repeats=3,
                  episode_length=500, airl_itrs=1000, msrd_itrs=10)
    dmsrd.rand_idx = list(range(len(demonstrations)))

    with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Order of Demos"])
        csvwriter.writerow(dmsrd.rand_idx)

    iteration = 0
    while iteration < len(dmsrd.rand_idx):
        dmsrd.new_demonstration(iteration)
        dmsrd.mixture_optimize(iteration)
        return_vals = dmsrd.msrd(iteration)
        iteration += 1

        with open(f'{dmsrd.log_path}/weights.csv', 'a') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows([["Weights"]])
            csvwriter.writerows(dmsrd.mixture_weights)
            # csvwriter.writerows([["Newmixes"]])
            # csvwriter.writerow(dmsrd.to_remove)
            csvwriter.writerows([["Randidx"]])
            csvwriter.writerow(dmsrd.rand_idx)
        if return_vals is not None:
            rew, div, like, corr, abl = return_vals
            with open(f'{dmsrd.log_path}/scale.csv', 'a') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                for i in range(len(rew)):
                    csvwriter.writerow([rew[i], div[i], like[i], corr[i], abl[i]])
            rewards.extend(rew)
            divergences.extend(div)
            likelihoods.extend(like)
            ground_truths.extend(corr)
            ablations.extend(abl)

    with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows([["Order of Demos"]])
        csvwriter.writerow(dmsrd.rand_idx)
        csvwriter.writerows([["Mixtures"]])
        csvwriter.writerows(dmsrd.mixture_weights)

    with open(f'{dmsrd.log_path}/final.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(rewards)
        csvwriter.writerow(divergences)
        csvwriter.writerow(likelihoods)
        csvwriter.writerow(ground_truths)
        csvwriter.writerow(ablations)


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
