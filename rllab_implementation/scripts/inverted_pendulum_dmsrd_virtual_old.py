import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.envs.base import TfEnv
from inverse_rl.models.fusion_manager import RamFusionDistr
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from inverse_rl.algos.dmsrd import DMSRD

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.dmsrd_virtual import new_likelihood, AIRLMultiStyleDynamic, Gaussian_Sum_Likelihood, ReLUModel, grid_objective, Grid_Search, MIP, GeometricMixturePolicies
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger


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
            policy = GaussianMLPPolicy(name=f'policy_{len(self.strategy_rewards)}', env_spec=self.env.spec,
                                       hidden_sizes=(32, 32))

            self.skill_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'skill_{len(self.strategy_rewards)}'))

            self.new_dictionary = {}
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{len(self.strategy_rewards)}')):
                self.new_dictionary[f'my_policy_{len(self.strategy_rewards)}_{idx}'] = var

            self.policies.append(policy)

            new_skill_reward = ReLUModel(f"skill_{len(self.strategy_rewards)}", self.env.observation_space.shape[0]) \
                if self.state_only \
                else ReLUModel(f"skill_{len(self.strategy_rewards)}",
                               self.env.observation_space.shape[0] + self.env.action_space.shape[0])
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{len(self.strategy_rewards)}')):
                self.save_dictionary[f'my_skill_{len(self.strategy_rewards)}_{idx}'] = var
            self.strategy_rewards.append(new_skill_reward)
            value_fn = ReLUModel(f"value_{len(self.experts_multi_styles)}", self.env.spec.observation_space.flat_dim)

            self.value_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'value_{len(self.experts_multi_styles)}'))
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'value_{len(self.experts_multi_styles)}')):
                self.save_dictionary[f'my_value_{len(self.experts_multi_styles)}_{idx}'] = var
            self.value_fs.append(value_fn)


        self.baselines.append(LinearFeatureBaseline(env_spec=self.env.spec))
        self.fusions.append(RamFusionDistr(10000, subsample_ratio=1))

        [c_i.append(0.0) for c_i in self.cluster_weights]
        self.cluster_weights.append([0.0] * len(self.strategy_rewards))
        self.cluster_weights[-1][-1] = 1.0
        np_cluster = np.array(self.cluster_weights)
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
        self.num_skills = len(self.experts_multi_styles)

        self.reward_fs = []
        self.algos = []
        for demo_ind in range(self.num_skills-1):
            with tf.variable_scope(f"iter_{iteration}_demo_{demo_ind}"):
                indices_to_train = self.demo_indices_to_train[demo_ind]
                irl_var_list = []
                for index in indices_to_train:
                    irl_var_list += self.skill_vars[index]
                    irl_var_list += self.value_vars[index]
                # if len(self.strategy_rewards) >= 3:
                #     irl_var_list += self.task_vars

                reward_weights = self.cluster_weights[demo_ind]
                irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                           np.array(self.strategy_rewards)[indices_to_train],
                                           np.array(self.value_fs)[indices_to_train],
                                           irl_var_list,
                                           np.array(reward_weights)[indices_to_train],
                                           expert_trajs=self.experts_multi_styles[demo_ind],
                                           state_only=self.state_only,
                                           new_strategy=True,
                                           fusion=self.fusions[demo_ind],
                                           l2_reg_skill=self.l2_reg_skill,
                                           l2_reg_task=self.l2_reg_task,
                                           max_itrs=self.discriminator_update_step)

                set_pol = self.policies[indices_to_train[0]]
                opt_pol = indices_to_train.shape[0] == 1
                if not opt_pol:
                    set_pol = GeometricMixturePolicies(self.env.spec, np.array(self.policies)[indices_to_train], np.array(reward_weights)[indices_to_train])
                algo = IRLTRPO(
                    reward_weights=reward_weights.copy(),
                    env=self.env,
                    policy=set_pol,
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
                    baseline=self.baselines[demo_ind],
                    optimize_pol=opt_pol
                )
                self.reward_fs.append(irl_model)
                self.algos.append(algo)

        demo_ind = self.num_skills - 1
        with tf.variable_scope(f"iter_{iteration}_skill_{demo_ind}"):

            indices_to_train = self.demo_indices_to_train[demo_ind]
            reward_weights = self.cluster_weights[demo_ind]
            irl_var_list = []
            for index in indices_to_train:
                irl_var_list += self.skill_vars[index]
                irl_var_list += self.value_vars[index]
            irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                              np.array(self.strategy_rewards)[indices_to_train],
                                              np.array(self.value_fs)[indices_to_train],
                                              irl_var_list,
                                              np.array(reward_weights)[indices_to_train],
                                              expert_trajs=self.experts_multi_styles[-1],
                                              state_only=self.state_only,
                                              fusion=self.fusions[demo_ind],
                                              new_strategy=True,
                                              l2_reg_skill=0.001,
                                              l2_reg_task=self.l2_reg_task,
                                              max_itrs=self.discriminator_update_step)

            algo = IRLTRPO(
                reward_weights=reward_weights.copy(),
                env=self.env,
                policy=self.policies[-1],
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
                baseline=self.baselines[demo_ind]
            )
            self.reward_fs.append(irl_model)
            self.algos.append(algo)

    def training_itr(self, iteration):
        # if iteration > 1:
        #     mutual_information_penalty = MIP(self.strategy_rewards[:-1], self.batch_size, self.mip_lr, self.env.spec,
        #                                      name=f"mip_{iteration}")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.save_dictionary)
            saver.restore(sess, f"{self.log_path}/model.ckpt")

            num_epochs = self.n_epochs #- 400
            # num_epochs = int(self.n_epochs / len(self.strategy_rewards))
            # if num_epochs < 100:
            #     num_epochs = 100
            if True:
                for epoch in range(num_epochs):
                    for skill in range(self.num_skills - 1):
                        with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
                            self.algos[skill].start_itr = self.repeat_each_epoch * epoch
                            self.algos[skill].n_itr = self.repeat_each_epoch * (epoch + 1)
                            self.algos[skill].train()
            else:
                for epoch in range(num_epochs):
                    task_reward_gradients = None
                    paths = []
                    for skill in range(self.num_skills-1):
                        with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
                            # for rew_ind in range(len(self.strategy_rewards)):
                            #     logger.record_tabular(f'Skill_{rew_ind}_weight',
                            #                           self.experts_multi_styles[skill][0]["reward_weights"][0, rew_ind])
                            self.algos[skill].start_itr = self.repeat_each_epoch * epoch
                            self.algos[skill].n_itr = self.repeat_each_epoch * (epoch + 1)
                            self.algos[skill].train()
                            # path = self.algos[skill].train()
                            # paths.append(path)
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

                # if iteration > 1:
                #     paths = list(itertools.chain(*paths))
                #     obs, acts = self.reward_fs[0].extract_paths(paths)
                #     rew_input = obs
                #     if not self.state_only:
                #         rew_input = tf.concat([obs, acts], axis=1)
                #     mip, _ = sess.run(
                #         [mutual_information_penalty.mutual_information_penalty, mutual_information_penalty.step],
                #         feed_dict={
                #             mutual_information_penalty.input: rew_input
                #         })

            for epoch in range(self.n_epochs):
                skill = self.num_skills - 1
                clust = len(self.strategy_rewards) - 1
                with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
                    # for rew_ind in range(len(self.strategy_rewards)):
                    #     logger.record_tabular(f'Skill_{rew_ind}_weight',
                    #                           self.experts_multi_styles[skill][0]["reward_weights"][0, rew_ind])
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


def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000,  repeat_each_skill=3,
                                                     separate_styles=True)

    dmsrd = DMSRDGAUSSIANMIX(env, demonstrations, log_prefix='inverted_pendulum_dmsrd_virtual')
    #dmsrd.rand_idx = [0, 3, 8, 5, 7, 2, 6, 9, 1, 4]
    dmsrd.rand_idx = [3, 7, 5, 6, 0, 8, 1, 9, 4, 2]
    dmsrd.n_epochs = 800

    iteration = 0

    while iteration < len(demonstrations):
        dmsrd.mixture_finding(iteration)

        dmsrd.build_graph(iteration)

        dmsrd.dmsrd_train(iteration)

        iteration += 1


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
