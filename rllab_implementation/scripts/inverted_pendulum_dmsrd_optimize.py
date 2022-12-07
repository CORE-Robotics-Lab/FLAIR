import numpy as np
import tensorflow as tf
import cma
import pyswarms.backend as P
from pyswarms.backend.topology import Star
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

import itertools
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from inverse_rl.algos.dmsrd_algo import DMSRD

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.dmsrd import AIRLMultiStyleDynamic
from inverse_rl.utils.divergence_utils import add_noise, query_tree
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger


class DMSRDOptimize(DMSRD):
    """
    Dynamic Multi Style Reward Distillation with different mixture optimization methods
    """

    def mixture_optimize(self, iteration):
        """
        Perform mixture optimization and determine if new strategy needs to be created for demonstration
        """
        self.num_skills = len(self.strategy_rewards)
        skill = self.num_skills - 1
        with tf.variable_scope(f"iter_{iteration}_new_skill"):
            irl_var_list = self.skill_vars[skill] + self.value_vars[skill]
            irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                              self.strategy_rewards[skill],
                                              self.value_fs[skill],
                                              expert_trajs=self.experts_multi_styles[-1],
                                              var_list=irl_var_list,
                                              state_only=self.state_only,
                                              mix_trajs=np.array(self.experts_multi_styles)[
                                                  np.arange(len(self.experts_multi_styles)) != skill],
                                              reward_weights=self.np_mixture_weights[:, skill][
                                                  np.arange(len(self.experts_multi_styles)) != skill],
                                              fusion=self.fusions[skill],
                                              new_strategy=True,
                                              l2_reg_skill=self.l2_reg_skill,
                                              l2_reg_task=self.l2_reg_task,
                                              max_itrs=self.discriminator_update_step)

            reward_weights = [0.0] * self.num_skills
            reward_weights[skill] = 1.0
            self.algo = IRLTRPO(
                reward_weights=reward_weights.copy(),
                env=self.env,
                policy=self.policies[skill],
                irl_model=irl_model,
                n_itr=1500,  # doesn't matter, will change
                batch_size=self.batch_size,
                max_path_length=self.episode_length,
                discount=0.99,
                store_paths=False,
                discrim_train_itrs=self.discriminator_update_step,
                irl_model_wt=1.0,
                entropy_weight=self.entropy_weight,  # from AIRL: this should be 1.0 but 0.1 seems to work better
                zero_environment_reward=True,
                baseline=self.baselines[skill]
            )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if iteration > 0:
                saver = tf.train.Saver(self.save_dictionary)
                saver.restore(sess, f"{self.log_path}/model.ckpt")

            if self.bool_save_vid:
                self.save_video(iteration)

            search_pols = np.array(self.policies[:-1])
            num_pols = len(search_pols)
            new_demo = self.experts_multi_styles[-1]
            if num_pols > 0:
                if num_pols > 1:
                    weights = np.random.uniform(0, 1, (self.grid_shots, num_pols))
                    weights = np.append(weights, np.diag(np.arange(num_pols)), axis=0)
                    for i in range(len(weights)):
                        if np.sum(weights[i]) <= 0:
                            weights[i] = np.ones_like(weights[i], dtype=np.float64)
                        weights[i] /= np.sum(weights[i], dtype=np.float64)
                else:
                    weights = np.array([[1.0]])

                weights_len = weights.shape[0]

                sampler = VectorizedSampler(self.algo, n_envs=weights_len)
                sampler.start_worker()

                # difference = np.zeros((1000, weights.shape[0]))
                difference = np.zeros((1000, weights_len, new_demo[0]["observations"][0].shape[0]))

                # rands = np.array([np.random.choice(num_pols, 1000, p=weight) for weight in weights])
                obses = sampler.vec_env.reset()
                actions = np.zeros((weights_len, 1))
                for timestep in range(1000):
                    actions[:weights_len] = np.einsum('ij,ijk->ik', weights, np.transpose(
                        [policy.get_actions(obses[:weights_len])[0] for policy in search_pols], (1, 0, 2)))
                    obses, rewards, dones, env_infos = sampler.vec_env.step(actions)
                    difference[timestep] = obses

                # distance = np.sqrt(np.sum(difference, axis=0))
                expert = new_demo[0]["observations"]
                n, d = expert.shape
                expert = add_noise(expert)
                m = 1000
                const = np.log(m) - np.log(n - 1)
                nn = query_tree(expert, expert, 3)
                distance = np.zeros(weights.shape[0])
                for idx, distribution in enumerate(np.transpose(difference, (1, 0, 2))):
                    nnp = query_tree(expert, distribution, 3 - 1)
                    distance[idx] = const + d * (np.log(nnp).mean() - np.log(nn).mean())
                best_idx = np.argmin(distance)
                best_distance = distance[best_idx]
                best_mix = weights[best_idx]
                self.grid_mixture = best_mix
                self.grid_distance = best_distance

                iters = 0
                if num_pols > 1:
                    # CMA Evolution Strategy
                    weight = np.random.uniform(0, 1.0, (num_pols))
                    if np.sum(weight) <= 0:
                        weight = np.ones_like(weight, dtype=np.float64)
                    weight /= np.sum(weight, dtype=np.float64)
                    es = cma.CMAEvolutionStrategy(weight, 1.0, {'bounds': [0, np.inf]})

                    curr_itr = 0
                    run_itrs = self.grid_shots // es.popsize
                    while curr_itr < run_itrs and not es.stop():
                        curr_itr += 1
                        weights = es.ask()
                        weights = np.array(weights)
                        for i in range(len(weights)):
                            if np.sum(weights[i]) <= 0:
                                weights[i] = np.ones_like(weights[i], dtype=np.float64)
                            weights[i] /= np.sum(weights[i], dtype=np.float64)
                        iters += es.popsize
                        weights_len = weights.shape[0]

                        sampler = VectorizedSampler(self.algo, n_envs=weights_len)
                        sampler.start_worker()

                        difference = np.zeros((1000, weights_len, new_demo[0]["observations"][0].shape[0]))

                        obses = sampler.vec_env.reset()
                        actions = np.zeros((weights_len, 1))
                        for timestep in range(1000):
                            actions = np.einsum('ij,ijk->ik', weights, np.transpose(
                                [policy.get_actions(obses[:weights_len])[0] for policy in search_pols], (1, 0, 2)))
                            obses, rewards, dones, env_infos = sampler.vec_env.step(actions)
                            difference[timestep] = obses
                        distance = np.zeros(weights.shape[0])
                        for idx, distribution in enumerate(np.transpose(difference, (1, 0, 2))):
                            nnp = query_tree(expert, distribution, 3 - 1)
                            distance[idx] = const + d * (np.log(nnp).mean() - np.log(nn).mean())

                        es.tell(weights, distance)
                    best_idx = np.argmin(distance)
                    best_distance = distance[best_idx]
                    best_mix = weights[best_idx]
                    self.gaussian_mixture = best_mix
                    self.gaussian_distance = best_distance

                    # PSO Optimization
                    my_topology = Star()  # The Topology Class
                    my_options = {'c1': 0.5, 'c2': 1.5, 'w': 0.6}  # w=inertia weight, c1=personal_best, c2=global_best
                    max_bounds = np.ones(num_pols)
                    min_bounds = np.zeros(num_pols)
                    bounds = (min_bounds, max_bounds)
                    my_swarm = P.create_swarm(n_particles=20, dimensions=num_pols, options=my_options,
                                              bounds=bounds)  # The Swarm Class

                    iterations = 100  # Set 100 iterations
                    for i in range(iterations):
                        # Part 1: Update personal best
                        my_swarm.current_cost = self.rollout(search_pols, const, d, nn, new_demo, expert,
                                                             my_swarm.position)  # Compute current cost
                        my_swarm.pbest_cost = \
                        self.rollout(search_pols, const, d, nn, new_demo, expert, my_swarm.pbest_pos)[
                            0]  # Compute personal best pos
                        my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm)  # Update and store

                        # Part 2: Update global best
                        # Note that gbest computation is dependent on your topology
                        if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
                            my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

                        # Part 3: Update position and velocity matrices
                        # Note that position and velocity updates are dependent on your topology
                        my_swarm.velocity = my_topology.compute_velocity(my_swarm)
                        my_swarm.position = self.norm_weights(my_topology.compute_position(my_swarm))

                    self.pso_distance = my_swarm.best_cost
                    self.pso_mixture = self.norm_weights([my_swarm.best_pos])[0]

                else:
                    self.gaussian_distance = self.grid_distance
                    self.gaussian_mixture = self.grid_mixture
                    self.pso_distance = self.grid_distance
                    self.pso_mixture = self.grid_mixture

                if self.gaussian_distance < self.grid_distance:
                    self.best_likelihood = self.gaussian_distance
                    self.best_mixture = self.gaussian_mixture
                else:
                    self.best_likelihood = self.grid_distance
                    self.best_mixture = self.grid_mixture

                self.new_dist = self.best_likelihood
                if self.best_likelihood > self.new_strategy_threshold:
                    skill = self.num_skills - 1
                    new_algo = self.algo
                    with rllab_logdir(algo=new_algo, dirname=self.log_path + '/skill_%d' % skill):
                        for epoch in range(self.airl_itrs):
                            new_algo.start_itr = self.repeat_each_epoch * epoch
                            new_algo.n_itr = self.repeat_each_epoch * (epoch + 1)
                            new_algo.train()
                    # New strategy
                    policy = self.policies[-1]
                    diff = np.zeros((self.episode_length, new_demo[0]["observations"][0].shape[0]))
                    obs = self.env.reset()
                    for timestep in range(self.episode_length):
                        act = policy.get_action(obs)[0]
                        obs, rewards, dones, env_infos = self.env.step(act)
                        diff[timestep] = obs
                    nnp = query_tree(expert, diff, 3 - 1)
                    self.new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())

                # Calculate new likelihood
                with rllab_logdir(algo=self.algo, dirname=self.log_path + '/mixture'):
                    logger.record_tabular(f'Demonstration', self.rand_idx[iteration])

                    logger.record_tabular(f'KL_Divergence', self.best_likelihood)
                    logger.record_tabular(f'KL_Mix', self.best_mixture)

                    logger.record_tabular(f'New_Distance', self.new_dist)

                    ratio = self.new_dist < self.best_likelihood
                    logger.record_tabular(f'Ratio', ratio)
                    logger.dump_tabular(with_prefix=False, write_header=True)

                    # Create new reward if below CRP parameter and add cluster weights
                    if not ratio or self.best_likelihood < self.new_strategy_threshold:
                        self.new_pol = False
                        [c_i.pop() for c_i in self.mixture_weights]
                        self.mixture_weights[-1] = (self.best_mixture / np.sum(self.best_mixture)).tolist()
                    else:
                        self.skill_to_demo[len(self.strategy_rewards)-1] = iteration
                        self.save_dictionary.update(self.new_dictionary)
                        self.new_pol = True
                if self.bool_save_vid:
                    self.save_mixture_video(self.policies[:-1], iteration)
            else:
                skill = self.num_skills - 1
                new_algo = self.algo
                with rllab_logdir(algo=new_algo, dirname=self.log_path + '/skill_%d' % skill):
                    for epoch in range(self.airl_itrs):
                        new_algo.start_itr = self.repeat_each_epoch * epoch
                        new_algo.n_itr = self.repeat_each_epoch * (epoch + 1)
                        new_algo.train()
                self.skill_to_demo[len(self.strategy_rewards)-1] = iteration
                self.save_dictionary.update(self.new_dictionary)
                self.new_pol = True

            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def rollout(self, search_pols, const, d, nn, new_demo, expert, weights):
        weights = np.array(weights)
        for i in range(len(weights)):
            if np.sum(weights[i]) <= 0:
                weights[i] = np.ones_like(weights[i], dtype=np.float64)
            weights[i] /= np.sum(weights[i], dtype=np.float64)
        weights_len = weights.shape[0]

        sampler = VectorizedSampler(self.algo, n_envs=weights_len)
        sampler.start_worker()

        difference = np.zeros((1000, weights_len, new_demo[0]["observations"][0].shape[0]))

        obses = sampler.vec_env.reset()
        actions = np.zeros((weights_len,1))
        for timestep in range(1000):
            actions = np.einsum('ij,ijk->ik', weights, np.transpose([policy.get_actions(obses[:weights_len])[0] for policy in search_pols], (1,0,2)))
            obses, rewards, dones, env_infos = sampler.vec_env.step(actions)
            difference[timestep] = obses
        distance = np.zeros(weights.shape[0])
        for idx, distribution in enumerate(np.transpose(difference, (1,0,2))):
            nnp = query_tree(expert, distribution, 3 - 1)
            distance[idx] = const + d * (np.log(nnp).mean() - np.log(nn).mean())

        return distance

    def norm_weights(self, weights):
        weights = np.array(weights)
        for i in range(len(weights)):
            if np.sum(weights[i]) <= 0:
                weights[i] = np.ones_like(weights[i], dtype=np.float64)
            weights[i] /= np.sum(weights[i], dtype=np.float64)
        return weights


def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000, repeat_each_skill=3,
                                                     separate_styles=True)
    # demonstrations = load_expert_from_core_MSD('data/100mix.pkl', length=1000, repeat_each_skill=3,
    #                                                  separate_styles=True)

    dmsrd = DMSRDOptimize(env, demonstrations, airl_itrs=0,msrd_itrs=0, grid_shots=2, log_prefix='inverted_pendulum_dmsrd_evolution')

    for iteration in range(len(demonstrations)):
        dmsrd.new_demonstration(iteration)
        dmsrd.mixture_optimize(iteration)
        dmsrd.msrd(iteration)


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
