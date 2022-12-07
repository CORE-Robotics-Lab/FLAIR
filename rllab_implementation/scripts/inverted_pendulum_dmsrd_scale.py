import numpy as np
import tensorflow as tf
import cma
import pyswarms.backend as P
from pyswarms.backend.topology import Star
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
import pickle

import itertools
from sandbox.rocky.tf.envs.base import TfEnv
from inverse_rl.models.fusion_manager import RamFusionDistr
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from inverse_rl.algos.dmsrd import DMSRD

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.dmsrd_enforce import new_likelihood, AIRLMultiStyleDynamic, Gaussian_Sum_Likelihood, ReLUModel, grid_objective, Grid_Search
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger
from scipy.spatial import cKDTree



# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_tree(x, xp, k):
    # https://github.com/BiuBiuBiLL/NPEET_LNC/blob/master/lnc.py
    # https://github.com/scipy/scipy/issues/9890 p=2 or np.inf
    tree = cKDTree(x)
    return tree.query(xp, k=k + 1, p=float('inf'))[0][:, k] # chebyshev distance of k+1-th nearest neighbor


class DMSRDGAUSSIANMIX(DMSRD):
    """
    Dynamic Multi Style Reward Distillation with mixing of action Gaussians
    """

    def mixture_finding(self, iteration):
        """
        Deriving mixture weights for new demonstration
        """
        self.experts_multi_styles.append(self.demonstrations[self.rand_idx[iteration]])

        if self.new_pol:
            new_strat_ind = len(self.strategy_rewards)

            new_skill_reward = ReLUModel(f"skill_{new_strat_ind}", self.env.observation_space.shape[0]) \
                if self.state_only \
                else ReLUModel(f"skill_{new_strat_ind}",
                               self.env.observation_space.shape[0] + self.env.action_space.shape[0])

            policy = GaussianMLPPolicy(name=f'policy_{new_strat_ind}', env_spec=self.env.spec,
                                       hidden_sizes=(32, 32))

            self.skill_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'skill_{new_strat_ind}'))

            self.new_dictionary = {}
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{new_strat_ind}')):
                self.new_dictionary[f'my_skill_{new_strat_ind}_{idx}'] = var
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{new_strat_ind}')):
                self.new_dictionary[f'my_policy_{new_strat_ind}_{idx}'] = var

            self.policies.append(policy)
            self.strategy_rewards.append(new_skill_reward)

            value_fn = ReLUModel(f"value_{new_strat_ind}", self.env.spec.observation_space.flat_dim)

            self.value_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'value_{new_strat_ind}'))
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'value_{new_strat_ind}')):
                self.new_dictionary[f'my_value_{new_strat_ind}_{idx}'] = var
            self.value_fs.append(value_fn)
            self.baselines.append(LinearFeatureBaseline(env_spec=self.env.spec))
            self.fusions.append(RamFusionDistr(10000, subsample_ratio=1))

        [c_i.append(0.0) for c_i in self.cluster_weights]
        self.cluster_weights.append([0.0] * len(self.strategy_rewards))
        self.cluster_weights[-1][-1] = 1.0
        self.np_cluster = np.array(self.cluster_weights)

        # optimization routine
        for idx, style in enumerate(self.experts_multi_styles):
            for path in style:
                path["reward_weights"] = np.repeat([self.cluster_weights[idx]], len(path["observations"]), axis=0)

    def build_graph(self, iteration):
        """
        Build DMSRD computation graph
        """
        self.num_skills = len(self.strategy_rewards)
        self.reward_fs = []
        self.algos = []
        for skill in range(self.num_skills - 1):
            with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
                irl_var_list = self.task_vars + self.skill_vars[skill] + self.value_vars[skill]
                skill_var_list = self.skill_vars[skill]
                irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                                  self.strategy_rewards[skill],
                                                  self.value_fs[skill],
                                                  skill_value_var_list=skill_var_list,
                                                  expert_trajs=self.experts_multi_styles[self.skill_to_demo[skill]],
                                                  mix_trajs=np.array(self.experts_multi_styles)[:-1][np.arange(len(self.experts_multi_styles)-1)!=self.skill_to_demo[skill]],
                                                  reward_weights=self.np_cluster[:-1, skill][np.arange(len(self.experts_multi_styles)-1)!=self.skill_to_demo[skill]],
                                                  var_list=irl_var_list,
                                                  state_only=self.state_only,
                                                  new_strategy=len(self.strategy_rewards) < 3,
                                                  fusion=self.fusions[skill],
                                                  l2_reg_skill=self.l2_reg_skill,
                                                  l2_reg_task=self.l2_reg_task,
                                                  max_itrs=self.discriminator_update_step)

                reward_weights = [0.0] * self.num_skills
                reward_weights[skill] = 1.0
                algo = IRLTRPO(
                    reward_weights=reward_weights.copy(),
                    env=self.env,
                    policy=self.policies[skill],
                    irl_model=irl_model,
                    n_itr=1500,  # doesn't matter, will change
                    batch_size=self.batch_size,
                    max_path_length=1000,
                    discount=0.99,
                    store_paths=False,
                    discrim_train_itrs=self.discriminator_update_step,
                    irl_model_wt=1.0,
                    entropy_weight=0.0,  # from AIRL: this should be 1.0 but 0.1 seems to work better
                    zero_environment_reward=True,
                    baseline=self.baselines[skill]
                )
                self.reward_fs.append(irl_model)
                self.algos.append(algo)

        skill = self.num_skills - 1
        with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
            irl_var_list = self.skill_vars[skill] + self.value_vars[skill]
            irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                              self.strategy_rewards[skill],
                                              self.value_fs[skill],
                                              expert_trajs=self.experts_multi_styles[-1],
                                              var_list=irl_var_list,
                                              state_only=self.state_only,
                                              mix_trajs=np.array(self.experts_multi_styles)[np.arange(len(self.experts_multi_styles))!=skill],
                                              reward_weights=self.np_cluster[:, skill][np.arange(len(self.experts_multi_styles))!=skill],
                                              fusion=self.fusions[skill],
                                              new_strategy=True,
                                              l2_reg_skill=0.01,
                                              l2_reg_task=self.l2_reg_task,
                                              max_itrs=self.discriminator_update_step)

            reward_weights = [0.0] * self.num_skills
            reward_weights[skill] = 1.0
            algo = IRLTRPO(
                reward_weights=reward_weights.copy(),
                env=self.env,
                policy=self.policies[skill],
                irl_model=irl_model,
                n_itr=1500,  # doesn't matter, will change
                batch_size=self.batch_size,
                max_path_length=1000,
                discount=0.99,
                store_paths=False,
                discrim_train_itrs=self.discriminator_update_step,
                irl_model_wt=1.0,
                entropy_weight=0.0,  # from AIRL: this should be 1.0 but 0.1 seems to work better
                zero_environment_reward=True,
                baseline=self.baselines[skill]
            )
            self.reward_fs.append(irl_model)
            self.algos.append(algo)

    def build_first_graph(self, iteration):
        """
        Build DMSRD computation graph
        """
        self.num_skills = len(self.strategy_rewards)
        self.reward_fs = []
        self.algos = []
        for skill in range(self.num_skills):
            with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
                irl_var_list = self.skill_vars[skill] + self.value_vars[skill]
                irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                                  self.strategy_rewards[skill],
                                                  self.value_fs[skill],
                                                  var_list=irl_var_list,
                                                  expert_trajs=self.experts_multi_styles[skill],
                                                  state_only=self.state_only,
                                                  fusion=self.fusions[skill],
                                                  new_strategy=True,
                                                  l2_reg_skill=0.0,
                                                  l2_reg_task=self.l2_reg_task,
                                                  max_itrs=self.discriminator_update_step)

                reward_weights = [0.0] * self.num_skills
                reward_weights[skill] = 1.0
                algo = IRLTRPO(
                    reward_weights=reward_weights.copy(),
                    env=self.env,
                    policy=self.policies[skill],
                    irl_model=irl_model,
                    n_itr=1500,  # doesn't matter, will change
                    batch_size=self.batch_size,
                    max_path_length=1000,
                    discount=0.99,
                    store_paths=False,
                    discrim_train_itrs=self.discriminator_update_step,
                    irl_model_wt=1.0,
                    entropy_weight=0.0,  # from AIRL: this should be 1.0 but 0.1 seems to work better
                    zero_environment_reward=True,
                    baseline=self.baselines[skill]
                )
                self.reward_fs.append(irl_model)
                self.algos.append(algo)

    def first_training_itr(self, iteration):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epochs):
                for skill in range(self.num_skills):
                    with rllab_logdir(algo=self.algos[skill], dirname=self.log_path+'/skill_%d' % skill):
                        self.algos[skill].start_itr = self.repeat_each_epoch*epoch
                        self.algos[skill].n_itr = self.repeat_each_epoch*(epoch+1)
                        self.algos[skill].train()

            self.save_video(iteration)

            search_pols = np.array(self.policies[:-1])
            num_pols = len(search_pols)
            new_demo = self.experts_multi_styles[-1]

            weights = np.array([[1.0]])

            weights_len = weights.shape[0]

            sampler = VectorizedSampler(self.algos[0], n_envs=weights_len)
            sampler.start_worker()

            # difference = np.zeros((1000, weights.shape[0]))
            difference = np.zeros((1000, weights_len, new_demo[0]["observations"][0].shape[0]))

            # rands = np.array([np.random.choice(num_pols, 1000, p=weight) for weight in weights])
            obses = sampler.vec_env.reset()
            actions = np.zeros((weights_len,1))
            for timestep in range(1000):
                actions[:weights_len] = np.einsum('ij,ijk->ik', weights, np.transpose([policy.get_actions(obses[:weights_len])[0] for policy in search_pols], (1,0,2)))
                # actions[weights_len:] = np.array([policy.get_action(obses[weights_len+idx])[0] for idx, policy in enumerate(search_pols[rands[:, timestep]])])
                obses, rewards, dones, env_infos = sampler.vec_env.step(actions)
                # difference[timestep] = obses - new_demo[0]["observations"][timestep]) +
                #                               np.square(obses - new_demo[1]["observations"][timestep]) +
                #                               np.square(obses - new_demo[2]["observations"][timestep]), axis=1)
                difference[timestep] = obses

            # distance = np.sqrt(np.sum(difference, axis=0))
            expert = new_demo[0]["observations"]
            n, d = expert.shape
            expert = add_noise(expert)
            m = 1000
            const = np.log(m) - np.log(n - 1)
            nn = query_tree(expert, expert, 3)
            distance = np.zeros(weights.shape[0])
            for idx, distribution in enumerate(np.transpose(difference, (1,0,2))):
                nnp = query_tree(expert, distribution, 3 - 1)
                distance[idx] = const + d * (np.log(nnp).mean() - np.log(nn).mean())
            best_idx = np.argmin(distance)
            # if best_idx < weights_len:
            #     type_mix = "geometric"
            # else:
            #     type_mix = "prob"
            best_distance = distance[best_idx]
            best_idx %= weights_len
            best_mix = weights[best_idx]
            self.grid_mixture = best_mix
            self.gaussian_mixture = best_mix
            self.best_likelihood = best_distance
            self.best_mixture = best_mix

            # New strategy
            policy = self.policies[-1]
            # diff = 0
            diff = np.zeros((1000, new_demo[0]["observations"][0].shape[0]))
            obs = self.env.reset()
            for timestep in range(1000):
                act = policy.get_action(obs)[0]
                obs, rewards, dones, env_infos = self.env.step(act)
                # diff += np.sum(np.square(obs - new_demo[0]["observations"][timestep]) +
                #                               np.square(obs - new_demo[1]["observations"][timestep]) +
                #                               np.square(obs - new_demo[2]["observations"][timestep]))
                diff[timestep] = obs
            # new_dist = np.sqrt(diff)
            nnp = query_tree(expert, diff, 3 - 1)
            new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())

            # Calculate new likelihood
            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/mixture'):
                logger.record_tabular(f'Demonstration', self.rand_idx[iteration])

                logger.record_tabular(f'KL_Divergence', self.best_likelihood)
                logger.record_tabular(f'KL_Mix', self.best_mixture)
                # logger.record_tabular(f'KL_Type', type_mix)

                logger.record_tabular(f'New_Distance', new_dist)

                ratio = new_dist < self.best_likelihood
                logger.record_tabular(f'Ratio', ratio)
                action_likelihood = np.array(
                    [self.reward_fs[0].eval_expert_probs(self.experts_multi_styles[-1], pol) for pol in
                     search_pols], dtype=np.float64)
                action_prob = np.array(
                    [self.reward_fs[0].eval_numerical_integral(self.experts_multi_styles[-1], pol) for pol in
                     search_pols], dtype=np.float64)

                logger.record_tabular(f'Max_LogProbs', [new_likelihood(prob) for prob in action_prob])
                logger.record_tabular(f'Best_LogProbs', grid_objective([self.best_mixture],
                                                                       np.resize(np.exp(action_prob), (
                                                                           action_prob.shape[0],
                                                                           action_prob.shape[1] *
                                                                           action_prob.shape[2]))))

                logger.record_tabular(f'Max_LogLikelihoods',
                                      [new_likelihood(likelihood) for likelihood in action_likelihood])
                logger.record_tabular(f'Best_LogLikelihood', grid_objective([self.best_mixture],
                                                                            np.resize(np.exp(action_likelihood),
                                                                                      (
                                                                                          action_likelihood.shape[
                                                                                              0],
                                                                                          action_likelihood.shape[
                                                                                              1] *
                                                                                          action_likelihood.shape[
                                                                                              2]))))
                logger.record_tabular(f'CRP', (self.crp_alpha / (self.crp_alpha + len(self.strategy_rewards))))
                logger.dump_tabular(with_prefix=False, write_header=True)

            self.skill_to_demo[0] = 0
            # Create new reward if below CRP parameter and add cluster weights
            if not ratio or self.best_likelihood < 0:
                self.new_pol = False
                [c_i.pop() for c_i in self.cluster_weights]
                self.cluster_weights[-1] = (self.best_mixture / np.sum(self.best_mixture)).tolist()
                self.demo_readjust.append(1)
            else:
                self.skill_to_demo[len(self.strategy_rewards)-1] = 1
                self.save_dictionary.update(self.new_dictionary)
                self.new_pol = True

            self.save_mixture_video(self.policies[:-1], iteration)
            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def training_itr(self, iteration):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.save_dictionary)
            saver.restore(sess, f"{self.log_path}/model.ckpt")

            num_epochs = self.n_epochs - 200
            # num_epochs = int(self.n_epochs / (len(self.strategy_rewards)-1))
            # if num_epochs < 1:
            #      num_epochs = 1
            if len(self.strategy_rewards) >= 3:
                for epoch in range(num_epochs):
                    task_reward_gradients = None
                    for skill in range(self.num_skills-1):
                        with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
                            self.algos[skill].start_itr = self.repeat_each_epoch * epoch
                            self.algos[skill].n_itr = self.repeat_each_epoch * (epoch + 1)
                            self.algos[skill].train()
                            if task_reward_gradients is None:
                                task_reward_gradients = self.algos[skill].center_reward_gradients
                            else:
                                assert task_reward_gradients.keys() == self.algos[skill].center_reward_gradients.keys()
                                for key in task_reward_gradients.keys():
                                    task_reward_gradients[key] += self.algos[skill].center_reward_gradients[key]

                    feed_dict = {}
                    assert self.task_reward.grad_map_vars.keys() == task_reward_gradients.keys()
                    for key in self.task_reward.grad_map_vars.keys():
                        feed_dict[self.task_reward.grad_map_vars[key]] = task_reward_gradients[key]
                    sess.run(self.task_reward.step, feed_dict=feed_dict)
            # else:
            #     for epoch in range(num_epochs):
            #         for skill in range(self.num_skills-1):
            #             with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
            #                 self.algos[skill].start_itr = self.repeat_each_epoch * epoch
            #                 self.algos[skill].n_itr = self.repeat_each_epoch * (epoch + 1)
            #                 self.algos[skill].train()

            # for epoch in range(self.n_epochs):
            #     skill = self.num_skills - 1
            #     with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
            #         self.algos[skill].start_itr = self.repeat_each_epoch * epoch
            #         self.algos[skill].n_itr = self.repeat_each_epoch * (epoch + 1)
            #         self.algos[skill].train()

            self.save_video(iteration)

            search_pols = np.array(self.policies[:-1])
            num_pols = len(search_pols)
            new_demo = self.experts_multi_styles[-1]

            expert = new_demo[0]["observations"]
            n, d = expert.shape
            expert = add_noise(expert)
            m = 1000
            const = np.log(m) - np.log(n - 1)
            nn = query_tree(expert, expert, 3)

            iters = 0
            if num_pols > 1:
                #PSO Optimization
                my_topology = Star()  # The Topology Class
                my_options = {'c1': 0.5, 'c2': 1.5, 'w': 0.6}  # w=inertia weight, c1=personal_best, c2=global_best
                max_bounds = np.ones(num_pols)
                min_bounds = np.zeros(num_pols)
                bounds = (min_bounds, max_bounds)
                my_swarm = P.create_swarm(n_particles=20, dimensions=num_pols, options=my_options, bounds=bounds)  # The Swarm Class

                iterations = 100  # Set 100 iterations
                for i in range(iterations):
                    # Part 1: Update personal best
                    my_swarm.current_cost = self.rollout(search_pols, const, d, nn, new_demo, expert, my_swarm.position)  # Compute current cost
                    my_swarm.pbest_cost = self.rollout(search_pols, const, d, nn, new_demo, expert, my_swarm.pbest_pos)[0]  # Compute personal best pos
                    my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm)  # Update and store

                    # Part 2: Update global best
                    # Note that gbest computation is dependent on your topology
                    if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
                        my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

                    # Part 3: Update position and velocity matrices
                    # Note that position and velocity updates are dependent on your topology
                    my_swarm.velocity = my_topology.compute_velocity(my_swarm)
                    my_swarm.position = self.norm_weights(my_topology.compute_position(my_swarm))

                self.best_likelihood = my_swarm.best_cost
                self.best_mixture = self.norm_weights([my_swarm.best_pos])[0]
                self.grid_mixture = self.norm_weights([my_swarm.best_pos])[0]
                self.gaussian_mixture = self.grid_mixture

            crp_alpha = (self.crp_alpha / (self.crp_alpha + len(self.strategy_rewards)))
            new_dist = self.best_likelihood
            if self.best_likelihood > 1:
                skill = self.num_skills - 1
                new_algo = self.algos[skill]
                with rllab_logdir(algo=new_algo, dirname=self.log_path + '/skill_%d' % skill):
                    for epoch in range(self.n_epochs):
                        new_algo.start_itr = self.repeat_each_epoch * epoch
                        new_algo.n_itr = self.repeat_each_epoch * (epoch + 1)
                        new_algo.train()
                # New strategy
                policy = self.policies[-1]
                diff = np.zeros((1000, new_demo[0]["observations"][0].shape[0]))
                obs = self.env.reset()
                for timestep in range(1000):
                    act = policy.get_action(obs)[0]
                    obs, rewards, dones, env_infos = self.env.step(act)
                    diff[timestep] = obs
                nnp = query_tree(expert, diff, 3 - 1)
                new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())

            # Calculate new likelihood
            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/mixture'):
                logger.record_tabular(f'Demonstration', self.rand_idx[iteration])

                logger.record_tabular(f'KL_Divergence', self.best_likelihood)
                logger.record_tabular(f'KL_Mix', self.best_mixture)

                logger.record_tabular(f'New_Distance', new_dist)

                ratio = new_dist < self.best_likelihood
                logger.record_tabular(f'Ratio', ratio)
                action_likelihood = np.array(
                    [self.reward_fs[0].eval_expert_probs(self.experts_multi_styles[-1], pol) for pol in
                     search_pols], dtype=np.float64)
                action_prob = np.array(
                    [self.reward_fs[0].eval_numerical_integral(self.experts_multi_styles[-1], pol) for pol in
                     search_pols], dtype=np.float64)

                logger.record_tabular(f'CRP', crp_alpha)
                logger.dump_tabular(with_prefix=False, write_header=True)

                # Create new reward if below CRP parameter and add cluster weights
                if not ratio or self.best_likelihood < 1:
                    self.new_pol = False
                    [c_i.pop() for c_i in self.cluster_weights]
                    self.cluster_weights[-1] = (self.best_mixture / np.sum(self.best_mixture)).tolist()
                    self.demo_readjust.append(iteration)
                else:
                    self.skill_to_demo[len(self.strategy_rewards)-1] = iteration
                    self.save_dictionary.update(self.new_dictionary)
                    self.new_pol = True

            self.save_mixture_video(self.policies[:-1], iteration)

            # Save mix
            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/mixture'):
                logger.record_tabular(f'Mixture', self.cluster_weights)
                logger.dump_tabular(with_prefix=False, write_header=True)

            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/probs'):
                for demo_ind in range(len(self.experts_multi_styles)):
                    pols = self.policies
                    if not self.new_pol:
                        pols = pols[:-1]
                    post_rew = get_reward(self.env, self.cluster_weights[demo_ind], pols)
                    divergence = get_divergence(self.env, self.cluster_weights[demo_ind], pols, self.experts_multi_styles[demo_ind])
                    logger.record_tabular(f"Demonstration {demo_ind}", post_rew)
                    logger.record_tabular(f"Divergence {demo_ind}", divergence)
                logger.dump_tabular(with_prefix=False, write_header=True)

            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/reward'):
                for traj in self.reward_trajectories:
                    reward_cent = tf.get_default_session().run(self.reward_fs[0].reward_task, feed_dict={self.reward_fs[0].obs_t: traj})
                    score = reward_cent[:, 0]
                    logger.record_tabular(f'Center', np.mean(score))
            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def rollout(self, search_pols, const, d, nn, new_demo, expert, weights):
        weights = np.array(weights)
        for i in range(len(weights)):
            weights[weights < 0] = 0
            if np.sum(weights[i]) <= 0:
                weights[i] = np.ones_like(weights[i], dtype=np.float64)
            weights[i] /= np.sum(weights[i], dtype=np.float64)
        weights_len = weights.shape[0]

        sampler = VectorizedSampler(self.algos[0], n_envs=weights_len)
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
            weights[weights < 0] = 0
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

    dmsrd = DMSRDGAUSSIANMIX(env, demonstrations, grid_shots=2000, log_prefix='inverted_pendulum_dmsrd_scale')
    #dmsrd.rand_idx = [0, 3, 8, 5, 7, 2, 6, 9, 1, 4]
    # dmsrd.rand_idx = [3, 7, 5, 6, 0, 8, 1, 9, 4, 2]

    trajectories = []
    for i in range(20):
        with open(f'data/trajs/trajectories_{i}.pkl', "rb") as f:
            trajectories.extend(pickle.load(f))

    dmsrd.reward_trajectories = trajectories

    dmsrd.n_epochs = 600

    iteration = 0
    dmsrd.mixture_finding(iteration)
    dmsrd.save_dictionary.update(dmsrd.new_dictionary)
    dmsrd.mixture_finding(iteration+1)
    dmsrd.build_first_graph(iteration)
    dmsrd.dmsrd_train(iteration)
    iteration += 2

    while iteration < len(demonstrations):
        dmsrd.mixture_finding(iteration)

        dmsrd.build_graph(iteration)

        dmsrd.dmsrd_train(iteration)

        iteration += 1


def get_reward(env, mixture, policies):
    episode_return = 0.0
    for _ in range(10):
        ob = env.reset()
        for timestep in range(1000):
            act = np.dot(mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            episode_return += rew
    return episode_return/10


def get_divergence(env, mixture, policies, demonstration):
    dist = 0.0
    for _ in range(10):
        obs = []
        ob = env.reset()
        for timestep in range(1000):
            act = np.dot(mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            obs.append(ob)

        expert = demonstration[0]["observations"]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 3)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    return dist/10


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
