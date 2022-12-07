#!/usr/bin/env python3

import gym
from garage.envs import GymEnv

from global_utils.utils import *
from airl.dmsrd_scale import DMSRDScale
import csv
from garage.experiment.deterministic import set_seed


def main():
    set_seed(0)
    demonstrations = load_expert_from_core_MSD(
        'data/LunarLander100mix.pkl', length=500,
        repeat_each_skill=3,
        separate_styles=True, squeeze_option=3)
    env = GymEnv('LunarLanderContinuous-v2', max_episode_length=500)
    env_test = gym.make('LunarLanderContinuous-v2')

    rewards, divergences, likelihoods, ground_truths, ablations = [], [], [], [], []

    dmsrd = DMSRDScale(env, env_test, demonstrations, trajs_path='lunar_trajs', reward_path='LunarTestReward',
                  bool_save_vid=False, log_prefix='lunar_lander_dmsrd_scale', new_strategy_threshold=40.0,
                  n_workers_irl=1, batch_when_finding_mixture=3, grid_shots=900, mix_repeats=3,
                  episode_length=500, airl_itrs=1000, msrd_itrs=300)
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
