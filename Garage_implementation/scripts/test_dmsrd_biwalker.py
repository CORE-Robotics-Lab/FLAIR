#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
import random
import itertools

import numpy as np
import tensorflow as tf
from models.fusion_manager import RamFusionDistr

from models.dmsrd_enforce import new_likelihood, AIRLMultiStyleDynamic, Gaussian_Sum_Likelihood, ReLUModel, grid_objective, Grid_Search
from datetime import datetime
import gym
import dowel
from dowel import logger, tabular
from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from airl.irl_trpo import TRPO

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
from garage.experiment.deterministic import set_seed
from airl.test_performance import *
import csv
from airl.dmsrd import DMSRD


def main():
    set_seed(0)
    demonstrations = load_expert_from_core_MSD(
        'data/BipedalWalker100mix.pkl', length=1000,
        repeat_each_skill=3,
        separate_styles=True, squeeze_option=3)[:10]
    env = GymEnv('BipedalWalker-v2', max_episode_length=1000)
    env_test = gym.make('BipedalWalker-v2')

    dmsrd = DMSRD(env, env_test, demonstrations, trajs_path='biwalker_trajs', reward_path='BipedalTestReward',
                  bool_save_vid=False, log_prefix='biwalker_dmsrd',
                  n_workers_irl=1, batch_when_finding_mixture=3, grid_shots=0, mix_repeats=3, start_msrd=2,
                  episode_length=1000, airl_itrs=0, msrd_itrs=0)

    dmsrd.log_path = 'data/biwalker_dmsrd/2022_05_24_10_59_57'
    mixture_weights = []
    pure_demonstrations = []
    with open(dmsrd.log_path + "/mixture.csv") as csv_file:
        reader = csv.reader(csv_file)
        # next(reader)
        # next(reader)
        next(reader)
        order_of_demo = next(reader)
        order_of_demo = list(map(lambda x: int(x), order_of_demo))
        # next(reader)
        # next(reader)
        next(reader)
        for i in range(len(demonstrations)):
            weight = next(reader)
            weight = list(map(lambda x: float(x), weight))
            mixture_weights.append(weight)
            for idx, each_weight in enumerate(weight):
                if each_weight == 1.0 and idx not in dmsrd.strategy_to_demo:
                    dmsrd.strategy_to_demo[len(pure_demonstrations)] = i
                    pure_demonstrations.append(i)

    with dmsrd._create_tf_session() as sess:
        dmsrd.rand_idx = order_of_demo
        for iteration in range(len(demonstrations)):
            is_new = iteration in pure_demonstrations
            dmsrd.new_pol = is_new
            dmsrd.new_demonstration(iteration)

            dmsrd.num_strategies = len(dmsrd.strategy_rewards)
            strategy = dmsrd.num_strategies - 1
            if is_new:
                dmsrd.policies = [GaussianMLPPolicy(name=f'policy_{strategy}',
                                                    env_spec=dmsrd.env.spec,
                                                    hidden_sizes=[32, 32]) for strategy in range(dmsrd.num_strategies)]
                for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{strategy}')):
                    dmsrd.new_dictionary[f'my_policy_{strategy}_{idx}'] = var
                dmsrd.save_dictionary.update(dmsrd.new_dictionary)

            if iteration == len(demonstrations) - 1:
                dmsrd.mixture_weights = mixture_weights
                dmsrd.new_pol = True  # to make sure the number of strategies is correctly calculated
                dmsrd.msrd(iteration, no_train=True)


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
