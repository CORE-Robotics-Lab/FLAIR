import numpy as np
import tensorflow as tf
import csv
import pickle

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import matplotlib.pyplot as plt

from inverse_rl.algos.dmsrd_algo import DMSRD
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.dmsrd_ablation import AIRLMultiStyleDynamicAblation
from inverse_rl.utils.divergence_utils import add_noise, query_tree
from inverse_rl.utils.eval_utils import get_rewdiv
from inverse_rl.utils.likelihood_utils import grid_objective
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.plot_utils import heatmap, annotate_heatmap
from global_utils.utils import *
import rllab.misc.logger as logger


class DMSRDAblationStudy(DMSRD):
    """
    Dynamic Multi Style Reward Distillation Ablation study for Between Class Discrimination
    """

    def mixture_optimize(self, iteration):
        self.num_skills = len(self.strategy_rewards)
        skill = self.num_skills - 1
        with tf.variable_scope(f"iter_{iteration}_new_skill"):
            irl_var_list = self.skill_vars[skill] + self.value_vars[skill]
            irl_model = AIRLMultiStyleDynamicAblation(self.env, self.task_reward,
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
                                              enforce=self.enforce,
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
                new_dist = self.best_likelihood
                if self.best_likelihood > 1:
                    skill = self.num_skills - 1
                    new_algo = algo
                    with rllab_logdir(algo=new_algo, dirname=self.log_path + '/skill_%d' % skill):
                        for epoch in range(self.airl_itrs):
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
                with rllab_logdir(algo=algo, dirname=self.log_path + '/mixture'):
                    logger.record_tabular(f'Demonstration', self.rand_idx[iteration])

                    logger.record_tabular(f'KL_Divergence', self.best_likelihood)
                    logger.record_tabular(f'KL_Mix', self.best_mixture)

                    logger.record_tabular(f'New_Distance', new_dist)

                    ratio = new_dist < self.best_likelihood
                    logger.record_tabular(f'Ratio', ratio)

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
                # self.save_mixture_video(self.policies[:-1], iteration)
            else:
                skill = self.num_skills - 1
                new_algo = algo
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
                    irl_model = AIRLMultiStyleDynamicAblation(self.env, self.task_reward,
                                                      self.strategy_rewards[skill],
                                                      self.value_fs[skill],
                                                      skill_value_var_list=skill_var_list,
                                                      expert_trajs=self.experts_multi_styles[self.skill_to_demo[skill]],
                                                      mix_trajs=np.array(self.experts_multi_styles)[np.arange(len(self.experts_multi_styles))!=self.skill_to_demo[skill]],
                                                      reward_weights=self.np_cluster[:, skill][np.arange(len(self.experts_multi_styles))!=self.skill_to_demo[skill]],
                                                      var_list=irl_var_list,
                                                      state_only=self.state_only,
                                                      enforce=self.enforce,
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
                for epoch in range(self.msrd_itrs):
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

    def key_metrics(self):
        if not self.new_pol:
            self.policies.pop()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.save_dictionary)
            saver.restore(sess, f"{self.log_path}/model.ckpt")

            task_rewards = []
            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/reward'):
                for traj in self.reward_trajectories:
                    reward_cent = tf.get_default_session().run(self.reward_fs[0].reward_task,
                                                               feed_dict={self.reward_fs[0].obs_t: traj})
                    score = reward_cent[:, 0]
                    task_rewards.append(np.mean(score))
                    logger.record_tabular(f'Center', np.mean(score))
                logger.record_tabular(f'Correlation', np.corrcoef(self.ground_truths, task_rewards))
                logger.dump_tabular(with_prefix=False, write_header=True)

            rew = []
            for demo in self.experts_multi_styles:
                strat_rew = []
                for strat in range(len(self.reward_fs)):
                    rew_repeat = 0
                    for traj in demo:
                        reward = tf.get_default_session().run(self.reward_fs[strat].reward_combined,
                                                              feed_dict={self.reward_fs[strat].obs_t: traj["observations"]})
                        rew_repeat += np.mean(reward[:, 0])
                    strat_rew.append(rew_repeat/len(demo))
                rew.append(strat_rew)

            rew = np.array(rew)
            for j in range(len(rew[0])):
                max_reward = np.amax(rew[:, j])
                rew[:, j] = np.exp(rew[:, j]-max_reward)

            name = [f'Demonstration {i}' for i in range(len(rew))]
            trajectories = [f'Strategy {i}' for i in range(len(rew[0]))]

            fig, ax = plt.subplots()

            im, cbar = heatmap(rew, name, trajectories, ax=ax,
                               cmap="YlGn", cbarlabel="reward")
            texts = annotate_heatmap(im)

            fig.tight_layout()
            plt.savefig(f'{self.log_path}/heatmap_{self.enforce}.png')
            plt.close()

            self.np_cluster = np.array(self.cluster_weights)
            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/ablation'):
                for j in range(len(self.reward_fs)):
                    a = self.np_cluster[:, j]
                    b = rew[:, j]
                    similiarity = np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
                    dist = 1. - similiarity
                    logger.record_tabular(f"Distance {j}", dist)
                logger.dump_tabular(with_prefix=False, write_header=True)

            with rllab_logdir(algo=self.algos[0], dirname=self.log_path + '/probs'):
                for demo_ind in range(len(self.experts_multi_styles)):
                    post_likelihoods = np.array([self.reward_fs[0].eval_expert_probs(self.experts_multi_styles[demo_ind], self.policies[i])
                                  for i in range(len(self.policies))])
                    logger.record_tabular(f'Demonstration Log Likelihood {demo_ind}', grid_objective([self.cluster_weights[demo_ind]],
                                                                            np.resize(np.exp(post_likelihoods),
                                                                                      (
                                                                                          post_likelihoods.shape[
                                                                                              0],
                                                                                          post_likelihoods.shape[
                                                                                              1] *
                                                                                          post_likelihoods.shape[
                                                                                              2]))))
                    post_rew, divergence = get_rewdiv(self.env, self.cluster_weights[demo_ind], self.policies, self.experts_multi_styles[demo_ind])
                    logger.record_tabular(f"Demonstration {demo_ind}", post_rew)
                    logger.record_tabular(f"Divergence {demo_ind}", divergence)
                logger.dump_tabular(with_prefix=False, write_header=True)

""" DMSRD Between Class Ablation Experiment to study success of Between Class Discrimination """
def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000, repeat_each_skill=3,
                                                     separate_styles=True)

    for j in range(2):
        dmsrd = DMSRDAblationStudy(env, demonstrations, grid_shots=2000, log_prefix='inverted_pendulum_dmsrd_ablation')
        dmsrd.rand_idx = [3, 7, 5, 6, 0, 8, 1, 9, 4, 2]

        trajectories = []
        for i in range(20):
            with open(f'data/trajs/trajectories_{i}.pkl', "rb") as f:
                trajectories.extend(pickle.load(f))

        ground_truths = []
        with open('data/GroundTruthReward.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            first = True
            for row in reader:
                if first:
                    first = False
                    continue
                ground_truths.append(float(row[0]))
        dmsrd.ground_truths = ground_truths

        dmsrd.reward_trajectories = trajectories

        dmsrd.enforce = j%2 == 0

        for iteration in range(len(demonstrations)):
            dmsrd.new_demonstration(iteration)
            dmsrd.mixture_optimize(iteration)
            dmsrd.msrd(iteration)

        dmsrd.key_metrics()

        tf.compat.v1.reset_default_graph()


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
