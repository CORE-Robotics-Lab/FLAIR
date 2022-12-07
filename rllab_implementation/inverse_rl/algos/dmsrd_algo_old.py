import os

import random
from datetime import datetime
import itertools

import numpy as np
import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from inverse_rl.models.fusion_manager import RamFusionDistr
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.dmsrd import ReLUModel, AIRLMultiStyleDynamic
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger
import matplotlib.pyplot as plt


class DMSRD:
    """
    Base Dynamic Multi Style Reward Distillation framework
    """
    def __init__(self, env, demonstrations, log_prefix, state_only=True, airl_itrs=600, msrd_itrs=400,
                 grid_shots=10000, batch_size=10000, discriminator_update_step=10, repeat_each_epoch=1,
                 l2_reg_skill=0.01, l2_reg_task=0.0001):
        """
        Hyperparameter and log initialization routine
        """
        self.env = env
        self.act_dim = self.env.spec.action_space.shape[0]
        self.demonstrations = demonstrations

        # Hyperparameters
        self.state_only = state_only
        self.airl_itrs = airl_itrs
        self.msrd_itrs = msrd_itrs
        self.grid_shots = grid_shots
        self.batch_size = batch_size
        self.l2_reg_skill = l2_reg_skill,
        self.l2_reg_task = l2_reg_task,
        self.discriminator_update_step = discriminator_update_step
        self.repeat_each_epoch = repeat_each_epoch

        now = datetime.now()
        self.log_path = f"data/{log_prefix}/{now.strftime('%d_%m_%Y_%H_%M_%S')}"
        assert not os.path.exists(self.log_path), "log path already exist! "

        self.task_reward = ReLUModel("task", env.observation_space.shape[0]) \
            if self.state_only \
            else ReLUModel("task", env.observation_space.shape[0] + env.action_space.shape[0])

        self.mixture_weights = []

        self.policies = []
        self.value_fs = []
        self.strategy_rewards = []

        self.baselines = []
        self.fusions = []
        self.algos = []
        self.reward_fs = []

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
        self.new_pol = True

        self.rand_idx = random.sample(range(len(self.demonstrations)), len(self.demonstrations))

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

        if len(self.cluster_weights) > 0:
            # optimization routine
            self.num_skills = len(self.policies) - 1
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(self.save_dictionary)
                saver.restore(sess, f"{self.log_path}/model.ckpt")

                # Log std is fixed during training for consistent comparison
                action_likelihood = np.array(
                    [self.reward_fs[0].eval_expert_probs(self.experts_multi_styles[-1], self.policies[i]) for i in
                     range(self.num_skills)], dtype=np.float64)
                action_prob = np.array(
                    [self.reward_fs[0].eval_numerical_integral(self.experts_multi_styles[-1], self.policies[i]) for
                     i in
                     range(self.num_skills)], dtype=np.float64)

                # Grid search (sample large amount of random points) used to find mixture
                self.grid_mixture, self.grid_likelihood = Grid_Search(action_prob, self.grid_shots)
                self.gaussian_mixture, self.gaussian_likelihood, mixture_prob = Gaussian_Sum_Likelihood(
                    self.policies[:-1], self.reward_fs[0], self.experts_multi_styles[-1], self.grid_shots)

                if self.grid_likelihood > self.gaussian_likelihood:
                    self.best_likelihood = self.grid_likelihood
                    self.best_mixture = self.grid_mixture
                else:
                    self.best_likelihood = self.gaussian_likelihood
                    self.best_mixture = self.gaussian_mixture

                # self.save_action(iteration)

                self.save_mixture_video(self.policies[:-1], iteration)

                with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/debug'):
                    logger.record_tabular(f'Action Prob count less 1e-37', len(np.where(action_prob < 1e-37)))
                    logger.record_tabular(f'Action Prob count less 1e-323', len(np.where(action_prob < 1e-323)))
                    logger.record_tabular(f'Mixture Prob count less 1e-37', len(np.where(mixture_prob < 1e-37)))
                    logger.record_tabular(f'Mixture Prob count less 1e-323', len(np.where(mixture_prob < 1e-323)))
                    logger.dump_tabular(with_prefix=False, write_header=True)

                with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/mixture'):
                    logger.record_tabular(f'Demonstration', self.rand_idx[iteration])

                    logger.record_tabular(f'Gaussian_Best_LogProb', self.gaussian_likelihood)
                    logger.record_tabular(f'Gaussian_Mix', self.gaussian_mixture)

                    logger.record_tabular(f'Grid_Best_LogProb', self.grid_likelihood)
                    logger.record_tabular(f'Grid_Mix', self.grid_mixture)

                    logger.record_tabular(f'Max_LogProbs', [new_likelihood(prob) for prob in action_prob])
                    logger.record_tabular(f'Best_LogProbs', grid_objective([self.best_mixture],
                                                                           np.resize(np.exp(action_prob), (
                                                                           action_prob.shape[0],
                                                                           action_prob.shape[1] * action_prob.shape[2]))))

                    logger.record_tabular(f'Max_LogLikelihoods',
                                          [new_likelihood(likelihood) for likelihood in action_likelihood])
                    logger.record_tabular(f'Best_LogLikelihood', grid_objective([self.best_mixture],
                                                                                np.resize(np.exp(action_likelihood), (
                                                                                action_likelihood.shape[0],
                                                                                action_likelihood.shape[1] *
                                                                                action_likelihood.shape[2]))))

                    logger.dump_tabular(with_prefix=False, write_header=True)

        [c_i.append(0.0) for c_i in self.cluster_weights]
        self.cluster_weights.append([0.0] * len(self.strategy_rewards))
        self.cluster_weights[-1][-1] = 1.0
        np_cluster = np.array(self.cluster_weights).transpose()
        self.demo_indices_to_train = []
        for row in np_cluster:
            self.demo_indices_to_train.append(np.where(row > 0)[0])

        rew_size = len(self.strategy_rewards)
        # Change weights for buffer paths
        for fusion in self.fusions:
            for path in fusion.buffer:
                path["reward_weights"] = np.pad(path["reward_weights"], ((0, 0), (0, rew_size - path["reward_weights"].shape[1])),
                           mode='constant', constant_values=0)

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
                irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                                  self.strategy_rewards,
                                                  self.value_fs[skill],
                                                  expert_trajs=list(itertools.chain(
                                                      *[self.experts_multi_styles[ind] for ind in
                                                        self.demo_indices_to_train[skill]])),
                                                  var_list=irl_var_list,
                                                  state_only=self.state_only,
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
                    entropy_weight=0.1,  # from AIRL: this should be 1.0 but 0.1 seems to work better
                    zero_environment_reward=True,
                    baseline=self.baselines[skill]
                )
                self.reward_fs.append(irl_model)
                self.algos.append(algo)

        skill = self.num_skills - 1
        with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
            irl_var_list = self.skill_vars[skill] + self.value_vars[skill]
            irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                              self.strategy_rewards,
                                              self.value_fs[skill],
                                              expert_trajs=self.experts_multi_styles[-1],
                                              var_list=irl_var_list,
                                              state_only=self.state_only,
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
                entropy_weight=0.1,  # from AIRL: this should be 1.0 but 0.1 seems to work better
                zero_environment_reward=True,
                baseline=self.baselines[skill]
            )
            self.reward_fs.append(irl_model)
            self.algos.append(algo)

    def dmsrd_train(self, iteration):
        """
        Train DMSRD routine
        """
        # Run MSRD
        if iteration > 0:
            self.training_itr(iteration)
        else:
            self.first_training_itr(iteration)

    def training_itr(self, iteration):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.save_dictionary)
            saver.restore(sess, f"{self.log_path}/model.ckpt")

            num_epochs = self.n_epochs
            # num_epochs = int(self.n_epochs / len(self.strategy_rewards))
            # if num_epochs < 100:
            #      num_epochs = 100
            for epoch in range(num_epochs):
                task_reward_gradients = None
                for skill in range(self.num_skills-1):
                    with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
                        # for idx, demo in enumerate(self.experts_multi_styles):
                        #     logger.record_tabular(f'Demo_{idx}_Skill_{skill}_weight',
                        #                           demo[0]["reward_weights"][0, skill])
                        self.algos[skill].start_itr = self.repeat_each_epoch * epoch
                        self.algos[skill].n_itr = self.repeat_each_epoch * (epoch + 1)
                        self.algos[skill].train()
                        if skill < self.num_skills-1:
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

            for epoch in range(self.n_epochs):
                skill = self.num_skills - 1
                clust = len(self.strategy_rewards) - 1
                with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
                    # for idx, demo in enumerate(self.experts_multi_styles):
                    #     logger.record_tabular(f'Demo_{idx}_Skill_{clust}_weight',
                    #                           demo[0]["reward_weights"][0, clust])
                    self.algos[skill].start_itr = self.repeat_each_epoch * epoch
                    self.algos[skill].n_itr = self.repeat_each_epoch * (epoch + 1)
                    self.algos[skill].train()

            self.save_video(iteration)

            post_likelihood = np.array(self.reward_fs[0].eval_expert_probs(self.experts_multi_styles[-1], self.policies[-1]), dtype=np.float64)
            post_prob = np.array(self.reward_fs[0].eval_numerical_integral(self.experts_multi_styles[-1], self.policies[-1]), dtype=np.float64)

            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/debug'):
                logger.record_tabular(f'New Prob count less 1e-37', len(np.where(post_prob < 1e-37)))
                logger.record_tabular(f'New Prob count less 1e-323', len(np.where(post_prob < 1e-323)))
                logger.dump_tabular(with_prefix=False, write_header=True)

            # Calculate new likelihood
            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/mixture'):
                new_strat_likelihood = new_likelihood(post_prob)
                logger.record_tabular(f'New_Prob', new_strat_likelihood)
                logger.record_tabular(f'New_Likelihood', new_likelihood(post_likelihood))

                ratio = new_strat_likelihood / self.best_likelihood
                logger.record_tabular(f'Ratio', ratio)
                logger.record_tabular(f'CRP', (self.crp_alpha / (self.crp_alpha + len(self.strategy_rewards))))

                # Create new reward if below CRP parameter and add cluster weights
                if ratio < (self.crp_alpha / (self.crp_alpha + len(self.strategy_rewards))):
                    self.save_dictionary.update(self.new_dictionary)
                    self.new_pol = True
                else:
                    self.new_pol = False
                    [c_i.pop() for c_i in self.cluster_weights]
                    self.cluster_weights[-1] = (self.best_mixture / np.sum(self.best_mixture)).tolist()
                logger.dump_tabular(with_prefix=False, write_header=True)

            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def first_training_itr(self, iteration):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.n_epochs):
                for skill in range(self.num_skills):
                    with rllab_logdir(algo=self.algos[skill], dirname=self.log_path+'/skill_%d' % skill):
                        for idx, demo in enumerate(self.experts_multi_styles):
                            logger.record_tabular(f'Demo_{idx}_Skill_{skill}_weight', demo[0]["reward_weights"][0, skill])
                        self.algos[skill].start_itr = self.repeat_each_epoch*epoch
                        self.algos[skill].n_itr = self.repeat_each_epoch*(epoch+1)
                        self.algos[skill].train()

            self.save_video(iteration)
            post_likelihood = np.array(self.reward_fs[0].eval_expert_probs(self.experts_multi_styles[-1], self.policies[-1]), dtype=np.float64)
            post_prob = np.array(self.reward_fs[0].eval_numerical_integral(self.experts_multi_styles[-1], self.policies[-1]), dtype=np.float64)

            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/debug'):
                logger.record_tabular(f'New Prob count less 1e-37', len(np.where(post_prob < 1e-37)))
                logger.record_tabular(f'New Prob count less 1e-323', len(np.where(post_prob < 1e-323)))
                logger.dump_tabular(with_prefix=False, write_header=True)

            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/mixture'):
                logger.record_tabular(f'DemonstrationOG', self.rand_idx[0])
                logger.record_tabular(f'New_Prob', new_likelihood(post_prob))
                logger.record_tabular(f'New_Likelihood', new_likelihood(post_likelihood))
                logger.dump_tabular(with_prefix=False, write_header=True)

            self.save_dictionary.update(self.new_dictionary)
            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def save_mixture_video(self, policies, iteration):
        # Save New Mixture
        rand = np.random.choice(len(policies), 1000, p=self.grid_mixture)
        imgs = []
        ob = self.env.reset()
        for timestep in range(1000):
            act = policies[rand[timestep]].get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = self.env.step(act_executed)
            imgs.append(self.env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/prob/new_mixture_probabalistic.mp4"))

        rand = np.random.choice(len(policies), 1000, p=self.gaussian_mixture)
        imgs = []
        ob = self.env.reset()
        for timestep in range(1000):
            act = policies[rand[timestep]].get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = self.env.step(act_executed)
            imgs.append(self.env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/prob/new_mixture_geometric.mp4"))

        imgs = []
        ob = self.env.reset()
        for timestep in range(1000):
            act = np.dot(self.grid_mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = self.env.step(act_executed)
            imgs.append(self.env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/geometric/new_mixture_probabalistic.mp4"))


        imgs = []
        ob = self.env.reset()
        for timestep in range(1000):
            act = np.dot(self.gaussian_mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = self.env.step(act_executed)
            imgs.append(self.env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/geometric/new_mixture_geometric.mp4"))

    def save_video(self, iteration):
        # Save Video
        for skill in range(len(self.policies)):
            ob = self.env.reset()
            imgs = []
            policy = self.policies[skill]
            for timestep in range(1000):
                act = policy.get_action(ob)
                act_executed = act[0]
                ob, rew, done, info = self.env.step(act_executed)
                imgs.append(self.env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/skill/skill_{skill}.mp4"))

        # Save Mixtures
        for cluster_idx in range(len(self.cluster_weights)):
            mix = self.cluster_weights[cluster_idx]
            imgs = []
            ob = self.env.reset()
            for timestep in range(1000):
                act = np.dot(mix, [policy.get_action(ob)[0] for policy in self.policies])
                act_executed = act
                ob, rew, done, info = self.env.step(act_executed)
                imgs.append(self.env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/geometric/mixture_{cluster_idx}.mp4"))

        # Save Mixtures
        for cluster_idx in range(len(self.cluster_weights)):
            rand = np.random.choice(len(self.policies), 1000, p=self.cluster_weights[cluster_idx])
            imgs = []
            ob = self.env.reset()
            for timestep in range(1000):
                act = self.policies[rand[timestep]].get_action(ob)
                act_executed = act[0]
                ob, rew, done, info = self.env.step(act_executed)
                imgs.append(self.env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/prob/mixture_{cluster_idx}.mp4"))

    def save_histogram(self, vals, folder, path):
        plt.hist(vals, 30, density=True, facecolor='g', alpha=0.75)

        plt.xlabel('Log Likelihood')
        plt.ylabel('Frequency')
        plt.title('Histogram of Log Likelihood')
        plt.grid(True)

        directory = f'{self.log_path}/histogram/{folder}/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(f'{self.log_path}/histogram/{folder}/{path}')
        plt.close()
