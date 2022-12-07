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
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot



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

    def mixture_optimize(self, iteration):
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
                                              reward_weights=self.np_cluster[:, skill][
                                                  np.arange(len(self.experts_multi_styles)) != skill],
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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if iteration > 0:
                saver = tf.train.Saver(self.save_dictionary)
                saver.restore(sess, f"{self.log_path}/model.ckpt")

            # self.save_video(iteration)

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

                sampler = VectorizedSampler(algo, n_envs=weights_len)
                sampler.start_worker()

                difference = np.zeros((1000, weights_len, new_demo[0]["observations"][0].shape[0]))

                obses = sampler.vec_env.reset()
                actions = np.zeros((weights_len,1))
                for timestep in range(1000):
                    actions = np.einsum('ij,ijk->ik', weights, np.transpose([policy.get_actions(obses)[0] for policy in search_pols], (1,0,2)))
                    obses, rewards, dones, env_infos = sampler.vec_env.step(actions)
                    difference[timestep] = obses

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
                best_distance = distance[best_idx]
                best_mix = weights[best_idx]
                self.best_likelihood = best_distance
                self.best_mixture = best_mix
                self.grid_mixture = best_mix
                self.gaussian_mixture = best_mix

                crp_alpha = (self.crp_alpha / (self.crp_alpha + len(self.strategy_rewards)))

                skill = self.num_skills - 1
                new_algo = algo
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
                self.new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())

                # Calculate new likelihood
                with rllab_logdir(algo=algo, dirname=self.log_path + '/mixture'):
                    logger.record_tabular(f'Demonstration', self.rand_idx[iteration])

                    logger.record_tabular(f'KL_Divergence', self.best_likelihood)
                    logger.record_tabular(f'KL_Mix', self.best_mixture)

                    logger.record_tabular(f'New_Distance', self.new_dist)

                    ratio = self.new_dist < self.best_likelihood
                    logger.record_tabular(f'Ratio', ratio)
                    action_likelihood = np.array(
                        [irl_model.eval_expert_probs(self.experts_multi_styles[-1], pol) for pol in
                         search_pols], dtype=np.float64)
                    self.best_loglikelihood = grid_objective([self.best_mixture],
                                                             np.resize(np.exp(action_likelihood),
                                                                       (
                                                                           action_likelihood.shape[
                                                                               0],
                                                                           action_likelihood.shape[
                                                                               1] *
                                                                           action_likelihood.shape[
                                                                               2])))
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
                # self.save_mixture_video(self.policies[:-1], iteration)
            else:
                skill = self.num_skills - 1
                new_algo = algo
                with rllab_logdir(algo=new_algo, dirname=self.log_path + '/skill_%d' % skill):
                    for epoch in range(self.n_epochs):
                        new_algo.start_itr = self.repeat_each_epoch * epoch
                        new_algo.n_itr = self.repeat_each_epoch * (epoch + 1)
                        new_algo.train()
                self.skill_to_demo[len(self.strategy_rewards)-1] = iteration
                self.save_dictionary.update(self.new_dictionary)
                self.new_pol = True

            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def msrd(self, iteration):
        if len(self.strategy_rewards) >= 3:
            self.num_skills = len(self.strategy_rewards) - 1
            if self.new_pol:
                self.num_skills += 1
            self.reward_fs = []
            self.algos = []
            for skill in range(self.num_skills):
                with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
                    irl_var_list = self.task_vars + self.skill_vars[skill] + self.value_vars[skill]
                    skill_var_list = self.skill_vars[skill]
                    irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                                      self.strategy_rewards[skill],
                                                      self.value_fs[skill],
                                                      skill_value_var_list=skill_var_list,
                                                      expert_trajs=self.experts_multi_styles[self.skill_to_demo[skill]],
                                                      mix_trajs=np.array(self.experts_multi_styles)[np.arange(len(self.experts_multi_styles))!=self.skill_to_demo[skill]],
                                                      reward_weights=self.np_cluster[:, skill][np.arange(len(self.experts_multi_styles))!=self.skill_to_demo[skill]],
                                                      var_list=irl_var_list,
                                                      state_only=self.state_only,
                                                      new_strategy=False,
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

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(self.save_dictionary)
                saver.restore(sess, f"{self.log_path}/model.ckpt")
                num_epochs = self.msrd_itrs
                for epoch in range(num_epochs):
                    task_reward_gradients = None
                    for skill in range(self.num_skills):
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

                saver = tf.train.Saver(self.save_dictionary)
                saver.save(sess, f"{self.log_path}/model.ckpt")


def main():
    ground_labels = []
    divergence = []
    for i in range(3):
        env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

        # load expert demonstrations from DIAYN dataset
        demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000, repeat_each_skill=3,
                                                         separate_styles=True)

        dmsrd = DMSRDGAUSSIANMIX(env, demonstrations, grid_shots=2000, log_prefix='inverted_pendulum_dmsrd_roc')
        # dmsrd.rand_idx = [3, 7, 5, 6, 0, 8, 1, 9, 4, 2]

        dmsrd.n_epochs = 600
        dmsrd.msrd_itrs = 400

        for iteration in range(len(demonstrations)):
            dmsrd.mixture_finding(iteration)
            dmsrd.mixture_optimize(iteration)
            dmsrd.msrd(iteration)
            if iteration != 0:
                if dmsrd.best_likelihood < dmsrd.new_dist:
                    ground_labels.append(1)
                else:
                    ground_labels.append(0)
                divergence.append(dmsrd.best_loglikelihood)

        tf.compat.v1.reset_default_graph()

    auc = roc_auc_score(ground_labels, divergence)

    with rllab_logdir(algo=dmsrd.algos[0], dirname=dmsrd.log_path + '/auc'):
        logger.record_tabular(f'AUC Score', auc)
        logger.dump_tabular(with_prefix=False, write_header=True)

    # calculate roc curves
    lr_fpr, lr_tpr, thresholds = roc_curve(ground_labels, divergence)
    lowest_iu = None
    ix = None

    for i in range(len(thresholds)):
        ths = thresholds[i]
        above_thresholds = np.array(divergence)
        for j in range(len(above_thresholds)):
            if above_thresholds[j] >= ths:
                above_thresholds[j] = 1
            else:
                above_thresholds[j] = 0

        cm1 = confusion_matrix(ground_labels, above_thresholds)
        sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])

        iu_score = np.abs(auc - sensitivity) + np.abs(auc - specificity)
        if lowest_iu is None or iu_score < lowest_iu:
            lowest_iu = iu_score
            ix = i

        # summarize scores
        with rllab_logdir(algo=dmsrd.algos[0], dirname=dmsrd.log_path + '/auc'):
            logger.record_tabular(f'Threshold', ths)
            logger.record_tabular(f'Sensitivity', sensitivity)
            logger.record_tabular(f'Specificity', specificity)
            logger.record_tabular(f'IU Score', iu_score)
            logger.record_tabular(f'F1 Score', f1_score(ground_labels, above_thresholds))
            logger.record_tabular(f'ground_labels', ground_labels)
            logger.record_tabular(f'Below', above_thresholds)
            logger.record_tabular(f'Likelihoods', divergence)
            logger.dump_tabular(with_prefix=False, write_header=True)

    pyplot.plot(lr_fpr, lr_tpr, marker='.', lw=2, label=f'ROC Curve (Area = {round(auc,2)})')
    pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label='No Skill')
    pyplot.scatter(lr_fpr[ix], lr_tpr[ix], marker='o', color='black', label=f'Best (IU Score = {round(lowest_iu,2)})')
    # axis ground_labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("DMSRD Receiver Operating Characteristic Curve")
    pyplot.legend(loc="lower right")
    pyplot.savefig(f'{dmsrd.log_path}/roc_curve.png')


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
