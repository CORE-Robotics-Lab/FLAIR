import numpy as np
import tensorflow as tf
import itertools
from sandbox.rocky.tf.envs.base import TfEnv
from inverse_rl.models.fusion_manager import RamFusionDistr
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import csv
import pickle
import pyswarms.backend as P
from pyswarms.backend.topology import Star

from inverse_rl.algos.dmsrd import DMSRD

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.dmsrd_enforce import new_likelihood, AIRLMultiStyleDynamic, Gaussian_Sum_Likelihood, ReLUModel, grid_objective, Grid_Search
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger
from scipy.spatial import cKDTree
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
        self.experts_multi_styles.append(self.demonstrations.pop(0))

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
                                                  new_strategy=len(self.strategy_rewards) < 4,
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

            # self.save_video(iteration)

            search_pols = np.array(self.policies[:-1])
            new_demo = self.experts_multi_styles[-1]

            weights = np.array([[1.0]])
            weights_len = weights.shape[0]

            sampler = VectorizedSampler(self.algos[0], n_envs=weights_len)
            sampler.start_worker()

            difference = np.zeros((1000, weights_len, new_demo[0]["observations"][0].shape[0]))

            # rands = np.array([np.random.choice(num_pols, 1000, p=weight) for weight in weights])
            obses = sampler.vec_env.reset()
            actions = np.zeros((weights_len, 1))
            for timestep in range(1000):
                actions[:weights_len] = np.einsum('ij,ijk->ik', weights, np.transpose(
                    [policy.get_actions(obses[:weights_len])[0] for policy in search_pols], (1, 0, 2)))
                obses, rewards, dones, env_infos = sampler.vec_env.step(actions)
                difference[timestep] = obses

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
            self.gaussian_mixture = best_mix
            self.best_likelihood = best_distance
            self.best_mixture = best_mix

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
                logger.record_tabular(f'KL_Divergence', self.best_likelihood)
                logger.record_tabular(f'KL_Mix', self.best_mixture)

                logger.record_tabular(f'New_Distance', new_dist)

                ratio = new_dist < self.best_likelihood
                logger.record_tabular(f'Ratio', ratio)
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

            # self.save_mixture_video(self.policies[:-1], iteration)
            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

            pols = self.policies
            if not self.new_pol:
                pols = pols[:-1]
            rewards = []
            divergences = []
            likelihoods = []
            for demonst in range(len(self.experts_multi_styles)):
                post_likelihoods = np.array([self.reward_fs[0].eval_expert_probs(self.experts_multi_styles[demonst], pols[i])
                              for i in range(len(pols))])
                r, d = get_rewdiv(self.env, self.cluster_weights[demonst], pols, self.experts_multi_styles[demonst])
                rewards.append(r)
                divergences.append(d)
                likelihoods.append(grid_objective([self.cluster_weights[demonst]], np.resize(np.exp(post_likelihoods), (
                                                                                      post_likelihoods.shape[
                                                                                          0],
                                                                                      post_likelihoods.shape[
                                                                                          1] *
                                                                                      post_likelihoods.shape[
                                                                                          2]))))
            return np.mean(rewards), np.mean(divergences), np.mean(likelihoods)

    def training_itr(self, iteration):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.save_dictionary)
            saver.restore(sess, f"{self.log_path}/model.ckpt")

            num_epochs = self.n_epochs - 200
            if len(self.strategy_rewards) >= 4:
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
            # self.save_video(iteration)

            search_pols = np.array(self.policies[:-1])
            num_pols = len(search_pols)
            new_demo = self.experts_multi_styles[-1]
            weights = np.random.uniform(0, 1, (self.grid_shots, num_pols))
            weights = np.append(weights, np.diag(np.arange(num_pols)), axis=0)
            for i in range(len(weights)):
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
                actions[:weights_len] = np.einsum('ij,ijk->ik', weights, np.transpose([policy.get_actions(obses[:weights_len])[0] for policy in search_pols], (1,0,2)))
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

            new_dist = self.best_likelihood
            if self.best_likelihood > 1:
                for epoch in range(self.n_epochs):
                    skill = self.num_skills - 1
                    with rllab_logdir(algo=self.algos[skill], dirname=self.log_path + '/skill_%d' % skill):
                        self.algos[skill].start_itr = self.repeat_each_epoch * epoch
                        self.algos[skill].n_itr = self.repeat_each_epoch * (epoch + 1)
                        self.algos[skill].train()
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
                logger.record_tabular(f'KL_Divergence', self.best_likelihood)
                logger.record_tabular(f'KL_Mix', self.best_mixture)

                logger.record_tabular(f'New_Distance', new_dist)

                ratio = new_dist < self.best_likelihood
                logger.record_tabular(f'Ratio', ratio)
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

            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

            pols = self.policies
            if not self.new_pol:
                pols = pols[:-1]
            rewards = []
            divergences = []
            likelihoods = []
            for demonst in range(len(self.experts_multi_styles)):
                post_likelihoods = np.array([self.reward_fs[0].eval_expert_probs(self.experts_multi_styles[demonst], pols[i])
                              for i in range(len(pols))])
                r, d = get_rewdiv(self.env, self.cluster_weights[demonst], pols, self.experts_multi_styles[demonst])
                rewards.append(r)
                divergences.append(d)
                likelihoods.append(grid_objective([self.cluster_weights[demonst]], np.resize(np.exp(post_likelihoods), (
                                                                                      post_likelihoods.shape[
                                                                                          0],
                                                                                      post_likelihoods.shape[
                                                                                          1] *
                                                                                      post_likelihoods.shape[
                                                                                          2]))))

            task_rewards = []
            for traj in self.reward_trajectories:
                reward_cent = tf.get_default_session().run(self.reward_fs[0].reward_task, feed_dict={self.reward_fs[0].obs_t: traj})
                score = reward_cent[:, 0]
                task_rewards.append(np.mean(score))

            return np.mean(rewards), np.mean(divergences), np.mean(likelihoods), np.corrcoef(self.ground_truths, task_rewards)[0,1]

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
        size = (weights_len,1)
        actions = np.zeros((weights_len,1))
        for timestep in range(1000):
            actions = np.einsum('ij,ijk->ik', weights, np.transpose([policy.get_actions(obses)[1]['mean'] for policy in search_pols], (1,0,2)))
            actions += np.random.normal(size=size) * 0.06284241094022028
            # actions = np.einsum('ij,ijk->ik', weights, np.transpose([policy.get_actions(obses[:weights_len])[0] for policy in search_pols], (1,0,2)))
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

    demonstrations = load_expert_from_core_MSD('data/100mix.pkl', length=1000, repeat_each_skill=3,
                                                     separate_styles=True)[:10]

    dmsrd = DMSRDGAUSSIANMIX(env, demonstrations, grid_shots=2000, log_prefix='inverted_pendulum_dmsrd_scale')

    trajectories = []
    for i in range(20):
        with open(f'data/trajs/trajectories_{i}.pkl', "rb") as f:
            trajectories.extend(pickle.load(f))

    dmsrd.reward_trajectories = trajectories

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

    dmsrd.n_epochs = 600

    iteration = 0
    dmsrd.mixture_finding(iteration)
    dmsrd.save_dictionary.update(dmsrd.new_dictionary)
    dmsrd.mixture_finding(iteration+1)
    dmsrd.build_first_graph(iteration)

    rewards, divergences, likelihoods, ground_truths = [], [], [], []

    rew, div, like = dmsrd.first_training_itr(iteration)
    rewards.append(rew)
    divergences.append(div)
    likelihoods.append(like)
    iteration += 2

    with rllab_logdir(algo=dmsrd.algos[0], dirname=dmsrd.log_path + '/scale'):
        logger.record_tabular(f'Policy Returns', rew)
        logger.record_tabular(f'Divergence', div)
        logger.record_tabular(f'Log Likelihood', like)
        logger.dump_tabular(with_prefix=False, write_header=True)

    while len(dmsrd.demonstrations) > 0:
        dmsrd.mixture_finding(iteration)

        dmsrd.build_graph(iteration)

        rew, div, like, corr = dmsrd.training_itr(iteration)
        rewards.append(rew)
        divergences.append(div)
        likelihoods.append(like)
        ground_truths.append(corr)

        iteration += 1

        with rllab_logdir(algo=dmsrd.algos[0], dirname=dmsrd.log_path + '/scale'):
            logger.record_tabular(f'Policy Returns', rew)
            logger.record_tabular(f'Divergence', div)
            logger.record_tabular(f'Log Likelihood', like)
            logger.record_tabular(f'Task Reward Correlation', corr)
            logger.dump_tabular(with_prefix=False, write_header=True)

    with rllab_logdir(algo=dmsrd.algos[0], dirname=dmsrd.log_path + '/scale'):
        logger.record_tabular(f'Policy Returns', rewards)
        logger.record_tabular(f'Divergence', divergences)
        logger.record_tabular(f'Log Likelihood', likelihoods)
        logger.record_tabular(f'Task Reward Correlation', ground_truths)
        logger.dump_tabular(with_prefix=False, write_header=True)

    # rewards, divergences, likelihoods, ground_truths = [-40,-20,-10], [2.3,1.0,0.3], [-40000,-30000,-10000], [0.6,0.8]

    indices = list(range(9))

    # pyplot.rcParams['axes.labelsize'] = 15
    # pyplot.rcParams['axes.labelweight'] = 'bold'
    # pyplot.rcParams['axes.linewidth'] = 1
    fig, axs = pyplot.subplots(2, 2)
    fig.suptitle('Key Metrics Scaled to 100 Demonstrations', fontweight='bold')

    axs[0, 0].plot(indices, rewards)
    axs[0, 0].set_xlim([0, 10])
    axs[0, 0].set_ylabel('Returns')
    axs[0, 0].set_xlabel('Demonstrations')
    axs[0, 0].set_title('Environment Return')
    axs[0, 1].plot(indices, divergences, 'tab:orange')
    axs[0, 1].set_xlim([0, 10])
    axs[0, 1].set_ylabel('Estimated Divergece')
    axs[0, 1].set_xlabel('Demonstrations')
    axs[0, 1].set_title('Estimated KL Divergence')
    axs[1, 0].plot(indices, likelihoods, 'tab:green')
    axs[1, 0].set_xlim([0, 10])
    axs[1, 0].set_ylabel('Log Likelihood')
    axs[1, 0].set_xlabel('Demonstrations')
    axs[1, 0].set_title('Demonstration Log Likelihood')
    axs[1, 1].plot(list(range(8)), ground_truths, 'tab:red')
    axs[1, 1].set_xlim([0, 10])
    axs[1, 1].set_ylabel('Correlation')
    axs[1, 1].set_xlabel('Demonstrations')
    axs[1, 1].set_title('Ground Truth Correlations')

    fig.tight_layout(pad=1.0)
    pyplot.savefig(f'data/scale.png')


def get_rewdiv(env, mixture, policies, demonstration):
    episode_return = 0.0
    dist = 0.0
    for _ in range(10):
        obs = []
        ob = env.reset()
        size = policies[0].get_action(ob)[1]['mean'].shape
        for timestep in range(1000):
            act = np.dot(mixture, [policy.get_action(ob)[1]['mean'] for policy in policies])
            act += np.random.normal(size=act.size) * 0.06284241094022028
            # act = np.dot(mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            episode_return += rew
            obs.append(ob)

        expert = demonstration[0]["observations"]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 3)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    return episode_return/10, dist/10


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
