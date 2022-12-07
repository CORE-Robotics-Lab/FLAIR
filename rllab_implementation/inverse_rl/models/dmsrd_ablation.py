import tensorflow as tf
import numpy as np

from inverse_rl.models.dmsrd import AIRLMultiStyleDynamic
from inverse_rl.utils import TrainingIterator
from sandbox.rocky.tf.spaces.box import Box


class AIRLMultiStyleDynamicAblation(AIRLMultiStyleDynamic):
    """
    DMSRD Between Class Discrimination Ablation Experiment

    Args:
        env: Gym environment
        task_reward: Task reward ReLU model
        strategy_reward: Individual strategy reward ReLU model
        value_fn: Value function ReLU model
        var_list: Tensorflow variable list for MSRD training
        skill_value_var_list: Tensorflow variable list for between class discrimination training
        expert_trajs: Expert demonstrations to be used in AIRL training
        mix_trajs: Mixture trajectories to be used in between class discrimination training
        fusion: Fusion model to store trajectories during AIRL
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        discount (float): Gamma discount factor
        state_only (bool): Fix the learned reward to only depend on state
        max_itrs (int): Number of training iterations to run per fit step
        name (str): Tensorflow variable scope name for model
        new_strategy (bool): Whether to include Task reward and Between class training in AIRL
        l2_reg_skill (float): Regularization on the strategy reward output
        l2_reg_task (float): Regularization on the task reward output
    """

    def __init__(self, env, task_reward,
                 strategy_reward,
                 value_fn,
                 var_list,
                 skill_value_var_list=None,
                 expert_trajs=None,
                 mix_trajs=None,
                 reward_weights=None,
                 reward_arch_args=None,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 enforce=True,
                 max_itrs=100,
                 fusion=None,
                 name='airl',
                 new_strategy=False,
                 l2_reg_skill=0.1,
                 l2_reg_task=0.001):
        super(AIRLMultiStyleDynamicAblation, self).__init__()
        env_spec = env.spec
        self.task_reward = task_reward
        self.strategy_reward = strategy_reward
        self.value_fn = value_fn

        self.var_list = var_list
        self.skill_value_var_list = skill_value_var_list
        if reward_arch_args is None:
            reward_arch_args = {}

        if fusion:
            self.fusion = fusion
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        self.score_discrim = score_discrim
        self.gamma = discount
        self.set_demos(expert_trajs)
        self.mix_trajs = mix_trajs
        self.reward_weights = reward_weights

        self.enforce = enforce
        self.new_strategy = new_strategy
        self.state_only = state_only
        self.max_itrs = max_itrs

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(tf.float32, [None, self.dO], name='nobs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')
            self.traj_truth = tf.placeholder(tf.float32, [None, self.dO], name='traj_truth')
            self.traj = tf.placeholder(tf.float32, [None, self.dO], name='traj')
            self.mixture_weight = tf.placeholder(tf.float32, (), name='mixture_weight')

            with tf.variable_scope('discrim') as dvs:
                rew_input = self.obs_t
                if not self.state_only:
                    rew_input = tf.concat([self.obs_t, self.act_t], axis=1)

                self.reward_task = self.task_reward(rew_input)
                self.reward_skill = self.strategy_reward(rew_input)
                if self.new_strategy:
                    self.reward = self.reward_skill
                else:
                    self.reward = self.reward_task + self.reward_skill
                self.reward_combined = self.reward_task + self.reward_skill

                self.reg_loss_skill = l2_reg_skill * tf.reduce_sum(tf.square(self.reward_skill))
                self.reg_loss_task = l2_reg_task * tf.reduce_sum(tf.square(self.reward_task))

                self.value_output = self.value_fn(self.obs_t)
                self.value_output_next = self.value_fn(self.nobs_t)

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma * self.value_output_next
                log_p_tau = self.reward + self.gamma * self.value_output_next - self.value_output

            log_q_tau = self.lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau - log_pq)
            cent_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))
            self.discriminator_predict = tf.cast(log_p_tau > log_q_tau, tf.float32)
            self.acc = tf.reduce_mean(self.discriminator_predict * self.labels +
                                      (1 - self.discriminator_predict) * (1 - self.labels))

            self.loss = cent_loss + self.reg_loss_skill + self.reg_loss_task
            tot_loss = self.loss

            # 2nd Process Discriminator
            self.rew_truth_strat = tf.reduce_mean(tf.stop_gradient(self.strategy_reward(self.traj_truth)))
            self.rew_traj_strat = tf.reduce_mean(self.strategy_reward(self.traj))

            self.rew_truth_task = tf.reduce_mean(tf.stop_gradient(self.task_reward(self.traj_truth)))
            self.rew_traj_task = tf.reduce_mean(tf.stop_gradient(self.task_reward(self.traj)))
            # self.reg_traj_truth = l2_reg_skill * self.rew_traj_truth
            # self.reg_traj = l2_reg_skill * self.rew_traj

            # self.loss_skill = tf.square(tf.exp(self.rew_traj_strat)-self.labels*tf.exp(self.rew_truth_strat))
            self.loss_skill = tf.square(tf.exp(self.rew_traj_strat+self.rew_traj_task)-self.mixture_weight*tf.exp(self.rew_truth_strat+self.rew_truth_task))

            # 1st Process MSRD Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.gradient_all = self.optimizer.compute_gradients(tot_loss, var_list=self.var_list)
            self.gradient_skill = []
            self.gradient_task_var_name = []
            self.gradient_task_value = []
            for grad, var in self.gradient_all:
                if 'task' in var.name:
                    self.gradient_task_var_name.append(var.name)
                    self.gradient_task_value.append(grad)
                else:
                    self.gradient_skill.append((grad, var))
            self.step = self.optimizer.apply_gradients(self.gradient_skill)

            # 2nd Process Punish Optimizer
            self.gradient_alL_skill = self.optimizer.compute_gradients(self.loss_skill, var_list=self.skill_value_var_list)
            self.step_skill = self.optimizer.apply_gradients(self.gradient_alL_skill)

            self._make_param_ops(_vs)
            self.center_reward_gradients = 0

    def fit(self, paths, policy=None, batch_size=32, logger=None, lr=1e-3, **kwargs):

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths + old_paths

        # eval samples under current policy
        self._compute_path_probs(paths, insert=True)

        # eval expert log probs under current policy
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._insert_next_state(paths)
        self._insert_next_state(self.expert_trajs)
        obs, obs_next, acts, acts_next, path_probs = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))

        self.center_reward_gradients = {}
        for var_name, value in zip(self.gradient_task_var_name, self.gradient_task_value):
            self.center_reward_gradients[var_name] = np.zeros(shape=value.shape)

        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs,
                                  batch_size=batch_size)

            # Build feed dict
            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(
                np.float32)
            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr
            }

            loss, reg_loss_peri, reg_loss_center, gradient_center_value, _ = \
                tf.get_default_session().run([self.loss, self.reg_loss_skill, self.reg_loss_task,
                                              self.gradient_task_value, self.step],
                                             feed_dict=feed_dict)
            for idx, var_name in enumerate(self.gradient_task_var_name):
                self.center_reward_gradients[var_name] += gradient_center_value[idx]
            it.record('loss', loss)
            it.record('reg_loss_peri', reg_loss_peri)
            it.record('reg_loss_center', reg_loss_center)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                mean_reg_loss_peri = it.pop_mean('reg_loss_peri')
                mean_reg_loss_center = it.pop_mean('reg_loss_center')
                print('\tLoss:%f' % mean_loss)
            if it.itr == self.max_itrs - 1:
                acc = tf.get_default_session().run(self.acc, feed_dict={self.act_t: act_batch, self.obs_t: obs_batch,
                                                                        self.nobs_t: nobs_batch,
                                                                        self.labels: labels, self.lprobs: lprobs_batch})
                logger.record_tabular('Discriminator_acc', acc)
                logger.record_tabular('RegLossPeri', mean_reg_loss_peri)
                logger.record_tabular('RegLossCenter', mean_reg_loss_center)

        if logger:
            logger.record_tabular('GCLDiscrimLoss', mean_loss)
            energy, reward_task, reward_skill, logZ, dtau = \
                tf.get_default_session().run([self.reward, self.reward_task, self.reward_skill,
                                              self.value_output, self.discrim_output],
                                             feed_dict={self.act_t: acts, self.obs_t: obs, self.nobs_t: obs_next,
                                                        self.lprobs: np.expand_dims(path_probs, axis=1),})
            energy = -energy
            logger.record_tabular('GCLLogZ', np.mean(logZ))
            logger.record_tabular('GCLAverageEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageRewardTask', np.mean(reward_task))
            logger.record_tabular('GCLAverageRewardSkill', np.mean(reward_skill))
            logger.record_tabular('GCLAverageLogPtau', np.mean(-energy - logZ))
            logger.record_tabular('GCLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('GCLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('GCLAverageDtau', np.mean(dtau))

            # expert_obs_next = np.r_[expert_obs_next, np.expand_dims(expert_obs_next[-1], axis=0)]
            energy, reward_task, reward_skill, logZ, dtau = \
                tf.get_default_session().run([self.reward, self.reward_task, self.reward_skill,
                                              self.value_output, self.discrim_output],
                                             feed_dict={self.act_t: expert_acts,
                                                        self.obs_t: expert_obs,
                                                        self.nobs_t: expert_obs_next,
                                                        self.lprobs: np.expand_dims(expert_probs,
                                                                                    axis=1)})
            energy = -energy
            logger.record_tabular('GCLAverageExpertEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageExpertRewardTask', np.mean(reward_task))
            logger.record_tabular('GCLAverageExpertRewardSkill', np.mean(reward_skill))
            logger.record_tabular('GCLAverageExpertLogPtau', np.mean(-energy - logZ))
            logger.record_tabular('GCLAverageExpertLogQtau', np.mean(expert_probs))
            logger.record_tabular('GCLMedianExpertLogQtau', np.median(expert_probs))
            logger.record_tabular('GCLAverageExpertDtau', np.mean(dtau))

        if not self.new_strategy and self.enforce:
            for it in TrainingIterator(self.max_itrs, heartbeat=5):
                traj_truth = self.expert_trajs[np.random.randint(0, len(self.expert_trajs))]["observations"]
                trajs_ind = np.random.randint(0, len(self.mix_trajs))
                traj = self.mix_trajs[trajs_ind][np.random.randint(0, len(self.mix_trajs[trajs_ind]))]["observations"]

                # Build feed dict
                weight = self.reward_weights[trajs_ind]

                feed_dict = {
                    self.traj_truth: traj_truth,
                    self.traj: traj,
                    self.mixture_weight: weight,
                    self.lr: lr/10,
                }

                loss, _ = \
                    tf.get_default_session().run([self.loss_skill, self.step_skill],
                                                 feed_dict=feed_dict)
                it.record('punishloss', loss)
                # it.record('reg_loss_peri', reg_loss_peri)
                if it.heartbeat:
                    print(it.itr_message())
                    mean_loss_punish = it.pop_mean('punishloss')
                    # mean_reg_loss_peri = it.pop_mean('reg_loss_peri')
                    print('\tPunish Loss:%f' % mean_loss_punish)
                if it.itr == self.max_itrs - 1:
                    # acc = tf.get_default_session().run(self.acc,
                    #                                    feed_dict={self.traj_truth: traj_truth, self.traj: traj, self.labels: labels, self.lr: lr})
                    # logger.record_tabular('Punish_discriminator_acc', acc)
                    logger.record_tabular('Punish_discriminator_loss', mean_loss_punish)
                    # logger.record_tabular('PunishLossPeri', mean_reg_loss_peri)

        return mean_loss