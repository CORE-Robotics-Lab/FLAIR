import time
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from inverse_rl.models.fusion_manager import RamFusionDistr
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.dmsrd import AIRLMultiStyleDynamic
from inverse_rl.models.relu import ReLUModel
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.divergence_utils import add_noise, query_tree
from global_utils.utils import *
import rllab.misc.logger as logger

def get_rewdiv(env, mixture, policies, demonstration):
    dist = 0.0
    episode_return = 0.0
    for _ in range(10):
        obs = []
        ob = env.reset()
        for timestep in range(1000):
            act = np.dot(mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            episode_return += rew
            obs.append(ob)

        expert = demonstration[:1000]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 13)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    return episode_return/10, dist/10

class DMSRD:
    """
    Base Dynamic Multi Style Reward Distillation framework
    """
    def __init__(self, env, demonstrations, log_prefix, bool_save_vid=False, state_only=True, airl_itrs=600,
                 msrd_itrs=400, episode_length=1000, new_strategy_threshold=1.0, start_msrd=3, mix_repeats=1,
                 grid_shots=10000, batch_size=10000, discriminator_update_step=10, repeat_each_epoch=1,
                 entropy_weight=0.0, l2_reg_skill=0.01, l2_reg_task=0.0001):
        """
        Hyperparameter and log initialization routine
        :param env: environment with which the training will happen
        :param demonstrations: expert demonstrations
        :type demonstrations: list (different strategies) of lists (trajectories of the same strategy) of dictionaries (including observations and actions)
        :param log_prefix: the log folder prefix
        :type log_prefix: str
        :param bool_save_vid: whether to save video
        :type bool_save_vid: bool
        :param state_only: whether to parameterize reward function only on state
        :type state_only: bool
        :param airl_itrs: how many iterations airl training takes
        :type airl_itrs: int
        :param msrd_itrs: how many iteration msrd training takes
        :type msrd_itrs: int
        :param episode_length: length of the env episode
        :type episode_length: int
        :param new_strategy_threshold: the threshold to determine whether to accept the mixture
        :type new_strategy_threshold: float
        :param start_msrd: number of strategies needed for MSRD to start
        :type start_msrd: int
        :param mix_repeats: number of repeats for trajectory generation for each mixture to evaluate metric
        :type mix_repeats: int
        :param grid_shots: number of different random mixture weights to try for mixture optimization
        :type grid_shots: int
        :param batch_size: batch size of timesteps from the environment (number of trajectories * length of trajectories)
        :type batch_size: int
        :param discriminator_update_step: number of steps for the discriminator to take each iteration
        :type discriminator_update_step: int
        :param repeat_each_epoch: number of MSRD training for each strategy reward before task reward update
        :type repeat_each_epoch: int
        :param entropy_weight: entropy weight of AIRL training
        :type entropy_weight: float
        :param l2_reg_skill: l2 regularization on skill variables (MSRD)
        :type l2_reg_skill: float
        :param l2_reg_task: l2 regularization on task variables (MSRD)
        :type l2_reg_task: float
        """
        self.env = env
        self.demonstrations = demonstrations

        self.bool_save_vid = bool_save_vid
        self.state_only = state_only

        # Hyperparameters
        self.airl_itrs = airl_itrs
        self.msrd_itrs = msrd_itrs

        self.episode_length = episode_length
        self.new_strategy_threshold = new_strategy_threshold
        self.start_msrd = start_msrd
        self.mix_repeats = mix_repeats  # TODO: not used?

        self.grid_shots = grid_shots
        self.batch_size = batch_size
        self.l2_reg_skill = l2_reg_skill,
        self.l2_reg_task = l2_reg_task,
        self.discriminator_update_step = discriminator_update_step
        self.repeat_each_epoch = repeat_each_epoch
        self.entropy_weight = entropy_weight

        now = datetime.now()
        self.log_path = f"data/{log_prefix}/{now.strftime('%d_%m_%Y_%H_%M_%S')}"
        assert not os.path.exists(self.log_path), "log path already exist! "

        # create task reward network
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
        self.best_mixture = []

        self.skill_vars = []
        self.value_vars = []
        self.save_dictionary = {}
        self.new_dictionary = {}
        self.task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task')
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='task')):
            self.save_dictionary[f'my_task_{idx}'] = var

        self.skill_to_demo = {}
        self.new_pol = True  # the first strategy has to be new
        self.num_skills = 0

        # randomize the order of the demonstrations
        self.rand_idx = random.sample(range(len(self.demonstrations)), len(self.demonstrations))

    def new_demonstration(self, iteration):
        """
        Create reward and policies to incorporate new demonstration into DMSRD model
        """
        self.experts_multi_styles.append(self.demonstrations[self.rand_idx[iteration]])

        if self.new_pol:
            # if new policy (the last strategy is a new policy), we create a new strategy network to train the new demonstration
            # otherwise, we just reuse the strategy network created last time and override the weights
            new_strategy_ind = len(self.strategy_rewards)

            new_skill_reward = ReLUModel(f"skill_{new_strategy_ind}", self.env.observation_space.shape[0]) \
                if self.state_only \
                else ReLUModel(f"skill_{new_strategy_ind}",
                               self.env.observation_space.shape[0] + self.env.action_space.shape[0])

            policy = GaussianMLPPolicy(name=f'policy_{new_strategy_ind}', env_spec=self.env.spec,
                                       hidden_sizes=(32, 32))

            self.skill_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'skill_{new_strategy_ind}'))

            self.new_dictionary = {}
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{new_strategy_ind}')):
                self.new_dictionary[f'my_skill_{new_strategy_ind}_{idx}'] = var
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{new_strategy_ind}')):
                self.new_dictionary[f'my_policy_{new_strategy_ind}_{idx}'] = var

            self.policies.append(policy)
            self.strategy_rewards.append(new_skill_reward)

            value_fn = ReLUModel(f"value_{new_strategy_ind}", self.env.spec.observation_space.flat_dim)

            self.value_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'value_{new_strategy_ind}'))
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'value_{new_strategy_ind}')):
                self.new_dictionary[f'my_value_{new_strategy_ind}_{idx}'] = var
            self.value_fs.append(value_fn)
            self.baselines.append(LinearFeatureBaseline(env_spec=self.env.spec))
            self.fusions.append(RamFusionDistr(10000, subsample_ratio=1))

        [c_i.append(0.0) for c_i in self.mixture_weights]  # all the previous demonstrations have zero weight on the new strategy
        # the new demonstration has pure weight on the newest strategy
        self.mixture_weights.append([0.0] * len(self.strategy_rewards))
        self.mixture_weights[-1][-1] = 1.0
        self.np_mixture_weights = np.array(self.mixture_weights)

        # update the expert demonstration weight information
        for idx, style in enumerate(self.experts_multi_styles):
            for path in style:
                path["reward_weights"] = np.repeat([self.mixture_weights[idx]], len(path["observations"]), axis=0)

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
                                              mix_trajs=np.array(self.experts_multi_styles)[  # This mix traj is wrong but doesn't matter - new_strategy=True will not do any BCD
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
            algo = IRLTRPO(
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
                # if there is at least one available policy, try mixture optimization
                samples = 0
                expert = np.concatenate([demo["observations"] for demo in new_demo])[:self.episode_length * self.mix_repeats]
                n, d = expert.shape
                expert = add_noise(expert)
                m = self.episode_length * self.mix_repeats
                const = np.log(m) - np.log(n - 1)
                nn = query_tree(expert, expert, 3)
                if num_pols > 1:
                    # if there are more than one policy, we do mixture optimization

                    # random weights
                    weights = np.random.uniform(0, 1, (self.grid_shots, num_pols))
                    # pure weights
                    weights = np.append(weights, np.diag(np.arange(num_pols)), axis=0)
                    # make sure all the weights are valid
                    for i in range(len(weights)):
                        if np.sum(weights[i]) <= 0:
                            weights[i] = np.ones_like(weights[i], dtype=np.float64)
                        weights[i] /= np.sum(weights[i], dtype=np.float64)

                    weights = np.repeat(weights, self.mix_repeats, axis=0)
                    weights_len = weights.shape[0]

                    # batch the evaluation
                    batch = 30
                    rounds = weights_len // batch

                    start = time.time()
                    margin = 0
                    rewards_sample = []
                    divergences_sample = []
                    rewards_time = []
                    divergences_time = []
                    for i in range(rounds):
                        samples += batch
                        sampler = VectorizedSampler(algo, n_envs=batch)
                        sampler.start_worker()

                        # TODO: repeat for each weight to reduce variance?

                        difference = np.zeros(
                            (self.episode_length, batch, new_demo[0]["observations"][0].shape[0]))

                        weights_batch = weights[i * batch:(i + 1) * batch]

                        obses = sampler.vec_env.reset()
                        for timestep in range(self.episode_length):
                            actions = np.einsum('ij,ijk->ik', weights_batch, np.transpose([policy.get_actions(obses)[0] for policy in search_pols], (1,0,2)))
                            obses, rewards, dones, env_infos = sampler.vec_env.step(actions)
                            difference[timestep] = obses

                        distance = np.zeros(batch//self.mix_repeats)
                        diff_trans = np.transpose(difference, (1, 0, 2))
                        for idx in range(batch//self.mix_repeats):
                            nnp = query_tree(expert, np.concatenate(diff_trans[idx*self.mix_repeats:(idx+1)*self.mix_repeats]), 3 - 1)
                            distance[idx] = const + d * (np.log(nnp).mean() - np.log(nn).mean())
                        best_idx = np.argmin(distance)
                        best_mixture_divergence = distance[best_idx]
                        best_mix = weights[best_idx]
                        rew, div = get_rewdiv(self.env, best_mix, search_pols, expert)
                        rewards_sample.append(rew)
                        divergences_sample.append(div)
                        if time.time() - start > margin:
                            rew, div = get_rewdiv(self.env, best_mix, search_pols, expert)
                            rewards_time.append(rew)
                            divergences_time.append(div)
                            margin += 500

                        if best_mixture_divergence <= self.new_strategy_threshold:
                            break
                    with rllab_logdir(algo=algo, dirname=self.log_path + f'/sample_{iteration}'):
                        logger.record_tabular(f'Rewards', rewards_sample)
                        logger.record_tabular(f'Divergences', divergences_sample)
                        logger.record_tabular(f'Rewards_Time', rewards_time)
                        logger.record_tabular(f'Divergences_Time', divergences_time)
                        logger.dump_tabular(with_prefix=False, write_header=True)
                else:
                    # if there is only one policy, the only possible mixture is [1.0]
                    # TODO: repeat to reduce variance?
                    policy = self.policies[0]
                    diff = np.zeros((self.episode_length * self.mix_repeats, new_demo[0]["observations"][0].shape[0]))
                    timestep = 0
                    for repeat in range(self.mix_repeats):
                        obs = self.env.reset()
                        for _ in range(self.episode_length):
                            act = policy.get_action(obs)[0]
                            obs, rewards, dones, env_infos = self.env.step(act)
                            diff[timestep] = obs
                            timestep += 1
                    nnp = query_tree(expert, diff, 3 - 1)
                    best_mixture_divergence = const + d * (np.log(nnp).mean() - np.log(nn).mean())
                    best_mix = np.array([1.0])

                self.best_mixture = best_mix
                # not separating geometric mixture and probabilistic mixture anymore
                self.grid_mixture = best_mix
                self.gaussian_mixture = best_mix

                new_strategy_divergence = 100000
                if best_mixture_divergence > self.new_strategy_threshold:
                    # we only train new strategy if the mixture is not good enough
                    skill = self.num_skills - 1
                    new_algo = algo
                    with rllab_logdir(algo=new_algo, dirname=self.log_path + '/skill_%d' % skill):
                        for epoch in range(self.airl_itrs):
                            new_algo.start_itr = self.repeat_each_epoch * epoch
                            new_algo.n_itr = self.repeat_each_epoch * (epoch + 1)
                            new_algo.train()
                    # New strategy
                    policy = self.policies[-1]
                    diff = np.zeros((self.episode_length * self.mix_repeats, new_demo[0]["observations"][0].shape[0]))
                    timestep = 0
                    for repeat in range(self.mix_repeats):
                        obs = self.env.reset()
                        for _ in range(self.episode_length):
                            act = policy.get_action(obs)[0]
                            obs, rewards, dones, env_infos = self.env.step(act)
                            diff[timestep] = obs
                            timestep += 1
                    nnp = query_tree(expert, diff, 3 - 1)
                    new_strategy_divergence = const + d * (np.log(nnp).mean() - np.log(nn).mean())

                # Calculate new likelihood
                with rllab_logdir(algo=algo, dirname=self.log_path + '/mixture'):
                    logger.record_tabular(f'Demonstration', self.rand_idx[iteration])

                    logger.record_tabular(f'KL_of_Mix', best_mixture_divergence)
                    logger.record_tabular(f'KL_Minimized_Mix', self.best_mixture)
                    logger.record_tabular(f'KL_of_New', new_strategy_divergence)
                    ratio = new_strategy_divergence < best_mixture_divergence
                    logger.record_tabular(f'Ratio', ratio)
                    logger.dump_tabular(with_prefix=False, write_header=True)

                    if not ratio or best_mixture_divergence < self.new_strategy_threshold:
                        # Do not create new strategy and use mixture if the mixture is good enough or better than training a new strategy
                        self.new_pol = False
                        [c_i.pop() for c_i in self.mixture_weights]  # revert the 0 appended for all the demonstrations when creating a new strategy
                        self.mixture_weights[-1] = (self.best_mixture / np.sum(self.best_mixture)).tolist()  # record the best mixture
                    else:
                        # Accept the new strategy into the model
                        self.skill_to_demo[len(self.strategy_rewards)-1] = iteration  # record the pure strategy demonstration
                        self.save_dictionary.update(self.new_dictionary)
                        self.new_pol = True
                if self.bool_save_vid:
                    self.save_mixture_video(self.policies[:-1], iteration)
            else:
                # there is no policy available, i.e., the very first iteration
                skill = self.num_skills - 1
                new_algo = algo
                with rllab_logdir(algo=new_algo, dirname=self.log_path + '/skill_%d' % skill):
                    for epoch in range(self.airl_itrs):
                        new_algo.start_itr = self.repeat_each_epoch * epoch
                        new_algo.n_itr = self.repeat_each_epoch * (epoch + 1)
                        new_algo.train()
                self.skill_to_demo[len(self.strategy_rewards)-1] = iteration  # record pure strategy
                self.save_dictionary.update(self.new_dictionary)
                self.new_pol = True

            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def msrd(self, iteration):
        if len(self.strategy_rewards) >= self.start_msrd:
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
                                                      reward_weights=self.np_mixture_weights[:, skill][np.arange(len(self.experts_multi_styles))!=self.skill_to_demo[skill]],
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
                        n_itr=1500,
                        batch_size=self.batch_size,
                        max_path_length=self.episode_length,
                        discount=0.99,
                        store_paths=False,
                        discrim_train_itrs=self.discriminator_update_step,
                        irl_model_wt=1.0,
                        entropy_weight=self.entropy_weight,
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

    def save_mixture_video(self, policies, iteration):
        # Save Probabilistic Mixture with Probabilistic Mixture Optimization
        rand = np.random.choice(len(policies), self.episode_length, p=self.grid_mixture)
        imgs = []
        ob = self.env.reset()
        for timestep in range(self.episode_length):
            act = policies[rand[timestep]].get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = self.env.step(act_executed)
            imgs.append(self.env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/prob/new_mixture_probabalistic.mp4"))

        # Save Probabilistic Mixture with Geometric Mixture Optimization
        rand = np.random.choice(len(policies), self.episode_length, p=self.gaussian_mixture)
        imgs = []
        ob = self.env.reset()
        for timestep in range(self.episode_length):
            act = policies[rand[timestep]].get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = self.env.step(act_executed)
            imgs.append(self.env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/prob/new_mixture_geometric.mp4"))

        # Save Geometric Mixture with Probabilistic Mixture Optimization
        imgs = []
        ob = self.env.reset()
        for timestep in range(self.episode_length):
            act = np.dot(self.grid_mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = self.env.step(act_executed)
            imgs.append(self.env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/geometric/new_mixture_probabalistic.mp4"))

        # Save Geometric Mixture with Geometric Mixture Optimization
        imgs = []
        ob = self.env.reset()
        for timestep in range(self.episode_length):
            act = np.dot(self.gaussian_mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = self.env.step(act_executed)
            imgs.append(self.env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/geometric/new_mixture_geometric.mp4"))

    def save_video(self, iteration):
        # Save Pure Policy Videos
        for skill in range(len(self.policies)):
            ob = self.env.reset()
            imgs = []
            policy = self.policies[skill]
            for timestep in range(self.episode_length):
                act = policy.get_action(ob)
                act_executed = act[0]
                ob, rew, done, info = self.env.step(act_executed)
                imgs.append(self.env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/skill/skill_{skill}.mp4"))

        # Save Geometric Mixture Videos
        for cluster_idx in range(len(self.mixture_weights)):
            mix = self.mixture_weights[cluster_idx]
            imgs = []
            ob = self.env.reset()
            for timestep in range(self.episode_length):
                act = np.dot(mix, [policy.get_action(ob)[0] for policy in self.policies])
                act_executed = act
                ob, rew, done, info = self.env.step(act_executed)
                imgs.append(self.env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/geometric/mixture_{cluster_idx}.mp4"))

        # Save Probabilistic Mixtures Videos
        for cluster_idx in range(len(self.mixture_weights)):
            rand = np.random.choice(len(self.policies), self.episode_length, p=self.mixture_weights[cluster_idx])
            imgs = []
            ob = self.env.reset()
            for timestep in range(self.episode_length):
                act = self.policies[rand[timestep]].get_action(ob)
                act_executed = act[0]
                ob, rew, done, info = self.env.step(act_executed)
                imgs.append(self.env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{self.log_path}/policy_videos_{iteration}/prob/mixture_{cluster_idx}.mp4"))
