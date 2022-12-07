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
from airl.test_performance import *
import csv


class MixturePolicy:
    def __init__(self, policies, weights):
        self.policies = policies
        self.weights = weights

    def get_action(self, obs):
        return np.dot(self.weights, [policy.get_action(obs)[0] for policy in self.policies]), {}

    def reset(self):
        None


class DMSRD:
    """
    Base Dynamic Multi Style Reward Distillation framework
    """
    def __init__(self, env, env_test, demonstrations, log_prefix, crp_alpha=50, state_only=True, n_epochs=1000, mixture_starts=2,
                 grid_shots=10000, batch_size=10000, mix_repeats=3, l2_reg_skill=0.01, episode_length=500, render_video=True,
                 l2_reg_task=0.0001, gen_update_step=1, discriminator_update_step=10, repeat_each_epoch=1):
        """
        Hyperparameter and log initialization routine
        """
        self.env = env
        self.env_test = env_test
        self.act_dim = self.env.spec.action_space.shape[0]
        self.demonstrations = demonstrations
        self.episode_length = episode_length

        # Hyperparameters
        self.crp_alpha = crp_alpha
        self.state_only = state_only
        self.n_epochs = n_epochs
        self.mixture_starts = mixture_starts
        self.grid_shots = grid_shots
        self.batch_size = batch_size
        self.render_video = render_video
        self.mix_repeats = mix_repeats
        self.l2_reg_skill = l2_reg_skill,
        self.l2_reg_task = l2_reg_task,
        self.gen_update_step = gen_update_step
        self.discriminator_update_step = discriminator_update_step
        self.repeat_each_epoch = repeat_each_epoch

        now = datetime.now()
        self.log_path = f"data/{log_prefix}/{now.strftime('%d_%m_%Y_%H_%M_%S')}"
        assert not os.path.exists(self.log_path), "log path already exist! "

        self.task_reward = ReLUModel("task", env.observation_space.shape[0]) \
            if self.state_only \
            else ReLUModel("task", env.observation_space.shape[0] + env.action_space.shape[0])

        self.weight_collection = [[]]
        self.cluster_weights = []

        self.policies = []
        self.value_fs = []
        self.strategy_rewards = []

        self.baselines = []
        self.fusions = []
        self.algos = []
        self.reward_fs = []
        self.trainers = []

        self.experts_multi_styles = []
        self.best_likelihood = None
        self.best_mixture = []

        self.skill_vars = []
        self.value_vars = []
        self.save_dictionary = {}
        self.new_dictionary = {}
        self.task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task')
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='task')):
            self.save_dictionary[f'my_task_{idx}'] = var

        self.skill_to_demo = {}
        self.demo_readjust = []
        self.new_pol = True

        self.rand_idx = random.sample(range(len(self.demonstrations)), len(self.demonstrations))

    def mixture_finding(self, iteration):
        """
        Deriving mixture weights for new demonstration
        """
        self.experts_multi_styles.append(self.demonstrations[self.rand_idx[iteration]])

        if self.new_pol:
            new_strat_ind = len(self.strategy_rewards)
            snapshotter = Snapshotter(f'{self.log_path}/skill_{new_strat_ind}')
            trainer = Trainer(snapshotter)

            new_skill_reward = ReLUModel(f"skill_{new_strat_ind}", self.env.observation_space.shape[0]) \
                if self.state_only \
                else ReLUModel(f"skill_{new_strat_ind}",
                               self.env.observation_space.shape[0] + self.env.action_space.shape[0])

            self.skill_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'skill_{new_strat_ind}'))

            self.new_dictionary = {}
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{new_strat_ind}')):
                self.new_dictionary[f'my_skill_{new_strat_ind}_{idx}'] = var
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{new_strat_ind}')):
                self.new_dictionary[f'my_policy_{new_strat_ind}_{idx}'] = var

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
            self.trainers.append(trainer)

        # optimization routine
        for idx, style in enumerate(self.experts_multi_styles):
            for path in style:
                path["reward_weights"] = np.repeat([self.cluster_weights[idx]], len(path["observations"]), axis=0)

    def build_graph(self, iteration):
        """
        Build DMSRD computation graph
        """

    def build_first_graph(self, iteration):
        """
        Build DMSRD computation graph
        """

    def dmsrd_train(self, iteration):
        """
        Train DMSRD routine
        """
        # Run MSRD
        if iteration > 0:
            self.training_itr(iteration)
        else:
            self.first_training_itr(iteration)

    def save_mixture_video(self, policies, iteration):
        imgs = []
        ob = self.env_test.reset()
        for timestep in range(self.episode_length):
            act = np.dot(self.gaussian_mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = self.env_test.step(act_executed)
            imgs.append(self.env_test.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/geometric/new_mixture_geometric.mp4"))

    def save_video(self, iteration):
        # Save Mixtures
        for cluster_idx in range(len(self.cluster_weights)):
            mix = self.cluster_weights[cluster_idx]
            imgs = []
            ob = self.env_test.reset()
            for timestep in range(self.episode_length):
                act = np.dot(mix, [policy.get_action(ob)[0] for policy in self.policies])
                act_executed = act
                ob, rew, done, info = self.env_test.step(act_executed)
                imgs.append(self.env_test.render('rgb_array'))
            save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/geometric/mixture_{cluster_idx}.mp4"))

    def first_training_itr(self, iteration):
        self.num_skills = len(self.strategy_rewards)
        self.reward_fs = []
        self.algos = []
        self.policies = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for skill in range(self.num_skills):
                with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
                    irl_var_list = self.skill_vars[skill] + self.value_vars[
                        skill]
                    irl_model = AIRLMultiStyleDynamic(self.env,
                                                      self.task_reward,
                                                      self.strategy_rewards[
                                                          skill],
                                                      self.value_fs[skill],
                                                      var_list=irl_var_list,
                                                      expert_trajs=
                                                      self.experts_multi_styles[
                                                          skill],
                                                      state_only=self.state_only,
                                                      fusion=self.fusions[
                                                          skill],
                                                      new_strategy=True,
                                                      l2_reg_skill=0.01,
                                                      l2_reg_task=self.l2_reg_task,
                                                      max_itrs=self.discriminator_update_step)

                    reward_weights = [0.0] * self.num_skills
                    reward_weights[skill] = 1.0

                    policy = GaussianMLPPolicy(name=f'policy_{skill}',
                                               env_spec=self.env.spec,
                                               hidden_sizes=(32, 32))
                    self.policies.append(policy)

                    sampler = None

                    algo = TRPO(env_spec=self.env.spec,
                                policy=policy,
                                baseline=self.baselines[skill],
                                index=skill,
                                center_grads=True,
                                sampler=sampler,
                                irl_model=irl_model,
                                generator_train_itrs=self.gen_update_step,
                                discrim_train_itrs=10,
                                discount=0.99,
                                max_kl_step=0.01)

                    self.trainers[skill].setup(algo, self.env)
                    self.reward_fs.append(irl_model)
                    self.algos.append(algo)
            sess.run(tf.global_variables_initializer())
            for skill in range(self.num_skills):
                logger.remove_all()
                tabular_log_file = os.path.join(self.trainers[skill]._snapshotter.snapshot_dir, 'progress.csv')
                logger.add_output(dowel.CsvOutput(tabular_log_file))
                logger.add_output(dowel.StdOutput())

                sampler = RaySampler(agents=self.policies[skill],
                                     envs=self.env,
                                     max_episode_length=self.episode_length,
                                     is_tf_worker=True)
                self.trainers[skill]._sampler = sampler
                self.trainers[skill]._start_worker()
                self.trainers[skill].train(n_epochs=self.n_epochs, batch_size=10000)
                self.trainers[skill]._shutdown_worker()

            if self.render_video:
                self.save_video(iteration)

            new_demo = self.experts_multi_styles[-1]

            expert = np.concatenate([demo["observations"] for demo in new_demo])
            n, d = expert.shape
            expert = add_noise(expert)
            m = self.episode_length * self.mix_repeats
            const = np.log(m) - np.log(n - 1)
            nn = query_tree(expert, expert, 3)

            policy = self.policies[0]
            diff = np.zeros((self.episode_length * self.mix_repeats,
                             new_demo[0]["observations"][0].shape[0]))
            timestep = 0
            for repeat in range(self.mix_repeats):
                obs = self.env_test.reset()
                for _ in range(self.episode_length):
                    act = policy.get_action(obs)[0]
                    obs, rewards, dones, env_infos = self.env_test.step(act)
                    diff[timestep] = obs
                    timestep += 1
            nnp = query_tree(expert, diff, 3 - 1)
            new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
            self.grid_mixture = [1.0]
            self.gaussian_mixture = [1.0]
            self.best_likelihood = new_dist
            self.best_mixture = [1.0]

            # New strategy
            policy = self.policies[-1]
            diff = np.zeros((self.episode_length*self.mix_repeats, new_demo[0]["observations"][0].shape[0]))
            timestep = 0
            for repeat in range(self.mix_repeats):
                obs = self.env_test.reset()
                for _ in range(self.episode_length):
                    act = policy.get_action(obs)[0]
                    obs, rewards, dones, env_infos = self.env_test.step(act)
                    diff[timestep] = obs
                    timestep += 1
            nnp = query_tree(expert, diff, 3 - 1)
            new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())

            ratio = new_dist < self.best_likelihood

            logger.remove_all()
            tabular_log_file = os.path.join(f'{self.log_path}/mixture',
                                            'progress.csv')
            logger.add_output(dowel.CsvOutput(tabular_log_file))
            logger.add_output(dowel.StdOutput())
            tabular.record(f'Demonstration', self.rand_idx[iteration])
            tabular.record(f'KL_Divergence', self.best_likelihood)
            tabular.record(f'New_Distance', new_dist)
            logger.log(tabular)
            logger.dump_all(iteration)
            tabular.clear()

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

            if self.render_video:
                self.save_mixture_video(self.policies[:-1], iteration)
            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def training_itr(self, iteration):
        self.num_skills = len(self.strategy_rewards)
        self.reward_fs = []
        self.algos = []
        self.policies = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for skill in range(self.num_skills - 1):
                policy = GaussianMLPPolicy(name=f'policy_{skill}',
                                           env_spec=self.env.spec,
                                           hidden_sizes=(32, 32))
                self.policies.append(policy)
                # with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
                #     irl_var_list = self.task_vars + self.skill_vars[skill] + \
                #                    self.value_vars[skill]
                #     skill_var_list = self.skill_vars[skill]
                #     irl_model = AIRLMultiStyleDynamic(self.env,
                #                                       self.task_reward,
                #                                       self.strategy_rewards[
                #                                           skill],
                #                                       self.value_fs[skill],
                #                                       skill_value_var_list=skill_var_list,
                #                                       expert_trajs=
                #                                       self.experts_multi_styles[
                #                                           self.skill_to_demo[
                #                                               skill]],
                #                                       mix_trajs=np.array(
                #                                           self.experts_multi_styles)[
                #                                                 :-1][np.arange(
                #                                           len(self.experts_multi_styles) - 1) !=
                #                                                      self.skill_to_demo[
                #                                                          skill]],
                #                                       var_list=irl_var_list,
                #                                       state_only=self.state_only,
                #                                       new_strategy=len(
                #                                           self.strategy_rewards) < 3,
                #                                       fusion=self.fusions[
                #                                           skill],
                #                                       l2_reg_skill=self.l2_reg_skill,
                #                                       l2_reg_task=self.l2_reg_task,
                #                                       max_itrs=self.discriminator_update_step)
                #
                #     reward_weights = [0.0] * self.num_skills
                #     reward_weights[skill] = 1.0
                #
                #     sampler = RaySampler(agents=policy,
                #                          envs=self.env,
                #                          max_episode_length=self.env.spec.max_episode_length,
                #                          is_tf_worker=True, n_workers=1)
                #
                #     algo = TRPO(env_spec=self.env.spec,
                #                 policy=policy,
                #                 baseline=self.baselines[skill],
                #                 index=skill,
                #                 center_grads=True,
                #                 sampler=sampler,
                #                 irl_model=irl_model,
                #                 generator_train_itrs=self.gen_update_step,
                #                 discrim_train_itrs=10,
                #                 discount=0.99,
                #                 max_kl_step=0.01)
                #
                #     self.trainers[skill].setup(algo, self.env)
                #     self.trainers[skill]._start_worker()
                #     self.reward_fs.append(irl_model)
                #     self.algos.append(algo)

            skill = self.num_skills - 1
            with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
                # irl_var_list = self.skill_vars[skill] + self.value_vars[skill]
                # irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                #                                   self.strategy_rewards[skill],
                #                                   self.value_fs[skill],
                #                                   expert_trajs=
                #                                   self.experts_multi_styles[
                #                                       -1],
                #                                   var_list=irl_var_list,
                #                                   state_only=self.state_only,
                #                                   fusion=self.fusions[skill],
                #                                   new_strategy=True,
                #                                   l2_reg_skill=0.01,
                #                                   l2_reg_task=self.l2_reg_task,
                #                                   max_itrs=self.discriminator_update_step)
                #
                # reward_weights = [0.0] * self.num_skills
                # reward_weights[skill] = 1.0

                policy = GaussianMLPPolicy(name=f'policy_{skill}',
                                           env_spec=self.env.spec,
                                           hidden_sizes=(32, 32))
                self.policies.append(policy)
                sampler = None

                # algo = TRPO(env_spec=self.env.spec,
                #             policy=policy,
                #             baseline=self.baselines[skill],
                #             index=skill,
                #             center_grads=True,
                #             sampler=sampler,
                #             irl_model=irl_model,
                #             generator_train_itrs=self.gen_update_step,
                #             discrim_train_itrs=10,
                #             discount=0.99,
                #             max_kl_step=0.01)
                #
                # self.trainers[skill].setup(algo, self.env)
                # self.reward_fs.append(irl_model)
                # self.algos.append(algo)

            for skill in range(self.num_skills):
                for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope=f'policy_{skill}')):
                    self.save_dictionary[f'my_policy_{skill}_{idx}'] = var

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.save_dictionary)
            saver.restore(sess, f"{self.log_path}/model.ckpt")

            num_epochs = self.n_epochs - 500
            # num_epochs = int(self.n_epochs / (len(self.strategy_rewards)-1))
            # if num_epochs < 1:
            #      num_epochs = 1
            # if len(self.strategy_rewards) >= 3:
            #     for epoch in range(num_epochs):
            #         center_reward_gradients = None
            #         for skill in range(self.num_skills-1):
            #             trainer = self.trainers[skill]
            #             logger.remove_all()
            #             tabular_log_file = os.path.join(
            #                 trainer._snapshotter.snapshot_dir, 'progress.csv')
            #             logger.add_output(dowel.CsvOutput(tabular_log_file))
            #             logger.add_output(dowel.StdOutput())
            #             trainer.train(n_epochs=epoch + 1, batch_size=10000,
            #                           start_epoch=epoch)
            #             if center_reward_gradients is None:
            #                 center_reward_gradients = self.algos[
            #                     skill].center_reward_gradients
            #             else:
            #                 assert center_reward_gradients.keys() == self.algos[
            #                     skill].center_reward_gradients.keys()
            #                 for key in center_reward_gradients.keys():
            #                     center_reward_gradients[key] += \
            #                         self.algos[skill].center_reward_gradients[key]
            #         feed_dict = {}
            #         assert self.task_reward.grad_map_vars.keys() == center_reward_gradients.keys()
            #         for key in self.task_reward.grad_map_vars.keys():
            #             feed_dict[self.task_reward.grad_map_vars[key]] = \
            #                 center_reward_gradients[key]
            #         sess.run(self.task_reward.step, feed_dict=feed_dict)
            #
            # self.trainers[0]._shutdown_worker()
            #
            # if self.render_video:
            #     self.save_video(iteration)
            #
            # search_pols = np.array(self.policies[:-1])
            # num_pols = len(search_pols)
            # new_demo = self.experts_multi_styles[-1]
            #
            # weights = np.random.uniform(0, 1, (self.grid_shots, num_pols))
            # # weights = np.append(weights, np.diag(np.arange(num_pols)), axis=0)
            # for i in range(len(weights)):
            #     if np.sum(weights[i]) <= 0:
            #         weights[i] = np.ones_like(weights[i], dtype=np.float64)
            #     weights[i] /= np.sum(weights[i], dtype=np.float64)
            #
            # weights = np.repeat(weights, self.mix_repeats,axis=0)
            # weights_len = weights.shape[0]
            # difference = np.zeros((weights_len, self.episode_length, new_demo[0]["observations"][0].shape[0]))
            #
            # for i in range(weights_len//20):
            #     sampler = RaySampler(agents=[MixturePolicy(search_pols, weight) for weight in weights[i:i+20]],
            #                          envs=self.env,
            #                          max_episode_length=self.env.spec.max_episode_length,
            #                          is_tf_worker=True,n_workers=20)
            #     sampler.start_worker()
            #     difference[i:i+20] = np.array([ep["observations"] for ep in sampler.obtain_one_episode(1).to_list()])
            #     sampler.shutdown_worker()
            #
            # expert = np.concatenate([demo["observations"] for demo in new_demo])
            # n, d = expert.shape
            # expert = add_noise(expert)
            # m = self.episode_length * self.mix_repeats
            # const = np.log(m) - np.log(n - 1)
            # for i in self.mixtures:
            #     new_demo = self.experts_multi_styles[i]
            #     expert = np.concatenate(
            #         [demo["observations"] for demo in new_demo])
            #     expert = add_noise(expert)
            #     nn = query_tree(expert, expert, 3)
            #     distance = np.zeros(weights_len//self.mix_repeats)
            #     for idx in range(weights_len//self.mix_repeats):
            #         nnp = query_tree(expert, np.concatenate(difference[idx*self.mix_repeats:(idx+1)*self.mix_repeats]), 3 - 1)
            #         distance[idx] = const + d * (np.log(nnp).mean() - np.log(nn).mean())
            #     best_idx = np.argmin(distance)
            #     best_distance = distance[best_idx]
            #     best_mix = weights[best_idx]
            #     self.cluster_weights[i] = np.pad((best_mix / np.sum(best_mix)), (0,1)).tolist()
            #
            # expert = np.concatenate([demo["observations"] for demo in new_demo])
            # n, d = expert.shape
            # expert = add_noise(expert)
            # nn = query_tree(expert, expert, 3)
            # distance = np.zeros(weights_len//self.mix_repeats)
            # for idx in range(weights_len//self.mix_repeats):
            #     nnp = query_tree(expert, np.concatenate(difference[idx*self.mix_repeats:(idx+1)*self.mix_repeats]), 3 - 1)
            #     distance[idx] = const + d * (np.log(nnp).mean() - np.log(nn).mean())
            # best_idx = np.argmin(distance)
            # best_distance = distance[best_idx]
            # best_mix = weights[best_idx*self.mix_repeats]
            # self.grid_mixture = best_mix
            # self.gaussian_mixture = best_mix
            # self.best_likelihood = best_distance
            # self.best_mixture = best_mix
            #
            # new_dist = self.best_likelihood
            # if self.best_likelihood > 1:
            #     skill = self.num_skills - 1
            #     logger.remove_all()
            #     tabular_log_file = os.path.join(self.trainers[skill]._snapshotter.snapshot_dir, 'progress.csv')
            #     logger.add_output(dowel.CsvOutput(tabular_log_file))
            #     logger.add_output(dowel.StdOutput())
            #     sampler = RaySampler(agents=self.policies[skill],
            #                          envs=self.env,
            #                          max_episode_length=self.env.spec.max_episode_length,
            #                          is_tf_worker=True)
            #     self.trainers[skill]._sampler = sampler
            #     self.trainers[skill]._start_worker()
            #     self.trainers[skill].train(n_epochs=self.n_epochs, batch_size=10000)
            #     self.trainers[skill]._shutdown_worker()
            #
            #     # New strategy
            #     policy = self.policies[-1]
            #     diff = np.zeros((self.episode_length*self.mix_repeats, new_demo[0]["observations"][0].shape[0]))
            #     timestep = 0
            #     for repeat in range(self.mix_repeats):
            #         obs = self.env_test.reset()
            #         for _ in range(self.episode_length):
            #             act = policy.get_action(obs)[0]
            #             obs, rewards, dones, env_infos = self.env_test.step(act)
            #             diff[timestep] = obs
            #             timestep += 1
            #     nnp = query_tree(expert, diff, 3 - 1)
            #     new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
            #
            #     logger.remove_all()
            #     tabular_log_file = os.path.join(f'{self.log_path}/mixture',
            #                                     'progress.csv')
            #     logger.add_output(dowel.CsvOutput(tabular_log_file))
            #     logger.add_output(dowel.StdOutput())
            #     tabular.record(f'Demonstration', self.rand_idx[iteration])
            #     tabular.record(f'KL_Divergence', self.best_likelihood)
            #     tabular.record(f'New_Distance', new_dist)
            #     logger.log(tabular)
            #     logger.dump_all(iteration)
            #     tabular.clear()
            #
            #     ratio = new_dist < self.best_likelihood
            #     # Create new reward if below CRP parameter and add cluster weights
            #     if not ratio or self.best_likelihood < 1:
            #         self.new_pol = False
            #         [c_i.pop() for c_i in self.cluster_weights]
            #         self.cluster_weights[-1] = (self.best_mixture / np.sum(self.best_mixture)).tolist()
            #         self.demo_readjust.append(iteration)
            #         self.policies = self.policies[:-1]
            #     else:
            #         self.skill_to_demo[len(self.strategy_rewards)-1] = iteration
            #         self.save_dictionary.update(self.new_dictionary)
            #         self.new_pol = True
            # saver = tf.train.Saver(self.save_dictionary)
            # saver.save(sess, f"{self.log_path}/model_10.ckpt")
            #
            # if self.render_video:
            #     self.save_mixture_video(self.policies[:-1], iteration)
            # self.save_video(iteration+1)

            env_test = gym.make('BipedalWalker-v2')
            for i in range(len(self.policies)):
                policy = self.policies[i]
                ob = env_test.reset()
                imgs = []
                rews = 0
                for timestep in range(1000):
                    ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
                    rews += rew
                    imgs.append(env_test.render('rgb_array'))
                print(rews)
                save_video(imgs, os.path.join(f"{self.log_path}/policy_videos/skill_{i}.avi"))

            for idx, mixture in enumerate(self.cluster_weights):
                ob = env_test.reset()
                imgs = []
                rews = 0
                for timestep in range(1000):
                    action = np.dot(mixture, [policy.get_action(ob)[0] for policy in self.policies])
                    ob, rew, done, info = env_test.step(action)
                    rews += rew
                    imgs.append(env_test.render('rgb_array'))
                print(rews)
                save_video(imgs, os.path.join(f"{self.log_path}/mixture_videos/mix_{idx}.avi"))
            # likelhi = get_dmsrd_likelihood(self.experts_multi_styles,
            #                                self.policies,
            #                                self.reward_fs[0],
            #                                self.cluster_weights)
            # rew, div = get_dmsrd_divergence(self.env_test,
            #                                 self.cluster_weights,
            #                                 self.policies,
            #                                 self.experts_multi_styles,
            #                                 self.episode_length)
            #
            # with open(f'{self.log_path}/likelihood.csv',
            #           'w') as csvfile:
            #     # creating a csv writer object
            #     csvwriter = csv.writer(csvfile)
            #     csvwriter.writerow(["Likelihoods"])
            #     csvwriter.writerow(likelhi)
            #     csvwriter.writerow(["Rewards"])
            #     csvwriter.writerow(rew)
            #     csvwriter.writerow(["Divergences"])
            #     csvwriter.writerow(div)
            #
            # trajectories = []
            # for i in range(40):
            #     with open(f'data/lunar_trajs/trajectories_{i}.pkl', "rb") as f:
            #         trajectories.extend(pickle.load(f))
            #
            # record = np.zeros(len(trajectories))
            # for tidx, traj in enumerate(trajectories):
            #     reward_cent = tf.get_default_session().run(
            #         self.reward_fs[0].reward_task,
            #         feed_dict={self.reward_fs[0].obs_t: traj["observations"]})
            #     score = reward_cent[:, 0]
            #     record[tidx] = np.mean(score)
            #
            # with open(f'{self.log_path}/reward.csv', 'w') as csvfile:
            #     # creating a csv writer object
            #     csvwriter = csv.writer(csvfile)
            #     csvwriter.writerow(["Task Reward"])
            #     csvwriter.writerow(record.tolist())
            #
            # rew = []
            # for demo in self.experts_multi_styles:
            #     strat_rew = []
            #     for strat in range(len(self.reward_fs)):
            #         rew_repeat = 0
            #         for traj in demo:
            #             reward = tf.get_default_session().run(
            #                 self.reward_fs[strat].reward_skill,
            #                 feed_dict={
            #                     self.reward_fs[strat].obs_t: traj["observations"]})
            #             score = np.mean(reward[:, 0])
            #             rew_repeat += np.mean(score)
            #         strat_rew.append(rew_repeat)
            #     rew.append(strat_rew)
            #
            # rew = np.array(rew)
            # for j in range(len(rew[0])):
            #     rew[:, j] = (rew[:, j] - np.min(rew[:, j])) / np.ptp(rew[:, j])
            #
            # rew_nomix = rew[list(self.skill_to_demo.keys())]
            # name = [f'Demonstration {i}' for i in list(self.skill_to_demo.keys())]
            # trajectories = [f'Strategy {i}' for i in range(len(rew[0]))]
            #
            # fig, ax = plt.subplots()
            #
            # im, cbar = heatmap(rew_nomix, name, trajectories, ax=ax,
            #                    cmap="YlGn", cbarlabel="reward")
            # texts = annotate_heatmap(im)
            #
            # fig.tight_layout()
            # plt.savefig(f'{self.log_path}/heatmap.png')
            # plt.close()
            # saver = tf.train.Saver(self.save_dictionary)
            # saver.save(sess, f"{self.log_path}/model_11.ckpt")

    def mixture(self, iteration):
        """
        Deriving mixture weights for new demonstration
        """
        self.experts_multi_styles.append(self.demonstrations[self.rand_idx[iteration]])

    def metrics(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.save_dictionary)
            saver.restore(sess, f"{self.log_path}/model.ckpt")


def main():
    demonstrations = load_expert_from_core_MSD(
        'data/BipedalWalker10skills.pkl', length=1000,
        repeat_each_skill=3,
        separate_styles=True)
    env = GymEnv('BipedalWalker-v2', max_episode_length=1000)
    env_test = gym.make('BipedalWalker-v2')

    dmsrd = DMSRD(env, env_test, demonstrations, grid_shots=2000, gen_update_step=2, render_video=False, log_prefix='biwalker_dmsrd')

    dmsrd.log_path = 'data/biwalker_dmsrd/22_01_2022_14_58_04'
    # dmsrd.rand_idx = [2,8,4,9,1,6,7,3,0,5]
    dmsrd.rand_idx = [6,9,0,2,4,3,5,1,8,7]

    dmsrd.n_epochs = 0
    js = {0:0, 1:1, 2:6, 3:8}
    dmsrd.mixtures = [2, 3, 4, 5, 7, 9]
    dmsrd.skill_to_demo = js

    weights = np.random.uniform(0, 1,
                                (6, 4))
    weights = np.append(weights, np.diag(np.arange(4)),
                        axis=0)
    for i in range(len(weights)):
        if np.sum(weights[i]) <= 0:
            weights[i] = np.ones_like(weights[i],
                                      dtype=np.float64)
        weights[i] /= np.sum(weights[i], dtype=np.float64)
    dmsrd.cluster_weights = weights
    iteration = 0
    dmsrd.mixture_finding(iteration)
    dmsrd.save_dictionary.update(dmsrd.new_dictionary)
    dmsrd.mixture_finding(iteration+1)
    iteration += 2

    with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Demonstrations"])
        csvwriter.writerow(dmsrd.rand_idx)

    while iteration < len(demonstrations):
        if iteration in dmsrd.mixtures:
            dmsrd.mixture(iteration)
        else:
            dmsrd.mixture_finding(iteration)
            if iteration < len(demonstrations)-1:
                dmsrd.save_dictionary.update(dmsrd.new_dictionary)

        iteration += 1

        # with open(f'{dmsrd.log_path}/mixture.csv', 'a') as csvfile:
        #     # creating a csv writer object
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow(["Mixtures"])
        #     csvwriter.writerows(dmsrd.cluster_weights)

    iteration -= 1
    dmsrd.build_graph(iteration)

    dmsrd.dmsrd_train(iteration)


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
