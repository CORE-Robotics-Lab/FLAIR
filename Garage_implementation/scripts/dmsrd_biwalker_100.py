#!/usr/bin/env python3

import gym
from garage.envs import GymEnv

from global_utils.utils import *
from airl.dmsrd_scale import DMSRDScale
import csv


def main():
    demonstrations = load_expert_from_core_MSD(
        'data/BipedalWalker100mix.pkl', length=1000,
        repeat_each_skill=3,
        separate_styles=True, squeeze_option=3)
    env = GymEnv('BipedalWalker-v2', max_episode_length=1000)
    env_test = gym.make('BipedalWalker-v2')

    rewards, divergences, likelihoods, ground_truths, ablations = [], [], [], [], []

    dmsrd = DMSRDScale(env, env_test, demonstrations, trajs_path='biwalker_trajs', reward_path='BipedalTestReward', n_workers_irl=1, batch_when_finding_mixture=1, grid_shots=300, bool_save_vid=False, log_prefix='biwalker_dmsrd_scale', episode_length=1000, airl_itrs=900, msrd_itrs=200, start_msrd=3, mix_repeats=1, new_strategy_threshold=15.0)

    dmsrd.rand_idx = list(range(len(demonstrations)))

    with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Order of Demos"])
        csvwriter.writerow(dmsrd.rand_idx)

    for iteration in range(len(demonstrations)):
        dmsrd.new_demonstration(iteration)
        dmsrd.mixture_optimize(iteration)
        return_vals = dmsrd.msrd(iteration)
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
