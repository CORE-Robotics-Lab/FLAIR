import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net
from inverse_rl.utils import TrainingIterator


class ReLUModel:
    def __init__(self, scope, input_dim, layers=2, dout=1, d_hidden=32):
        """
        ReLUModel here is used to construct task reward
        because we need to create resuable weights but not construct the actual computation

        :param scope:
        :param input_dim:
        :param layers:
        :param dout:
        :param d_hidden:
        """
        dX = input_dim
        self.layers = layers
        self.dout = dout
        self.d_hidden = d_hidden
        self.Ws = []
        self.bs = []
        with tf.variable_scope(scope):
            for i in range(layers):
                with tf.variable_scope("layer_%d" % i):
                    self.Ws.append(tf.get_variable('W', shape=(dX, d_hidden)))
                    self.bs.append(tf.get_variable('b', initializer=tf.constant(np.zeros(d_hidden).astype(np.float32))))
                dX = d_hidden
            with tf.variable_scope("layer_last"):
                self.Ws.append(tf.get_variable('W', shape=(d_hidden, dout)))
                self.bs.append(tf.get_variable('b', initializer=tf.constant(np.zeros(dout).astype(np.float32))))
        self.grad_and_vars = []
        self.grad_map_vars = {}
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
            ph = tf.placeholder(dtype=tf.float32, shape=var.shape)
            self.grad_and_vars.append((ph, var))
            self.grad_map_vars[var.name] = ph
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.step = self.optimizer.apply_gradients(self.grad_and_vars)

    def __call__(self, x):
        out = x
        for i in range(self.layers):
            out = tf.nn.relu(tf.matmul(out, self.Ws[i]) + self.bs[i])
        out = tf.matmul(out, self.Ws[self.layers]) + self.bs[self.layers]
        return out


class AIRLMultiStyle(SingleTimestepIRL):
    """


    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """

    def __init__(self, env, task_reward,
                 skill_1_reward,
                 skill_2_reward,
                 expert_trajs=None,
                 reward_arch=relu_net,
                 reward_arch_args=None,
                 value_fn_arch=relu_net,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 max_itrs=100,
                 fusion=False,
                 name='airl',
                 l2_reg_skill=0.01,
                 l2_reg_task=0.001):
        super(AIRLMultiStyle, self).__init__()
        env_spec = env.spec
        self.task_reward = task_reward
        self.skill_1_reward = skill_1_reward
        self.skill_2_reward = skill_2_reward
        if reward_arch_args is None:
            reward_arch_args = {}

        if fusion:
            self.fusion = RamFusionDistr(500, subsample_ratio=1)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        self.score_discrim = score_discrim
        self.gamma = discount
        assert value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only = state_only
        self.max_itrs = max_itrs

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(tf.float32, [None, self.dO], name='nobs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.weights = tf.placeholder(tf.float32, [None, 2], name="weights")
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            with tf.variable_scope('discrim') as dvs:
                rew_input = self.obs_t
                if not self.state_only:
                    rew_input = tf.concat([self.obs_t, self.act_t], axis=1)
                self.reward_task = self.task_reward(rew_input)
                self.reward_skill_1 = self.skill_1_reward(rew_input)
                self.reward_skill_2 = self.skill_2_reward(rew_input)
                self.reward_skill_weighted = self.reward_skill_1 * self.weights[:, 0:1] + self.reward_skill_2 * self.weights[:, 1:2]
                self.reward = self.reward_task + self.reward_skill_weighted
                # reg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dvs.name)
                # self.reg_loss = l2_reg * tf.reduce_sum([tf.reduce_sum(tf.square(var)) for var in reg_vars])
                self.reg_loss_skill = l2_reg_skill * tf.reduce_sum(tf.square(self.reward_skill_1) * self.weights[:, 0:1] + tf.square(self.reward_skill_2) * self.weights[:, 1:2])
                self.reg_loss_task = l2_reg_task * tf.reduce_sum(tf.square(self.reward_task))

                # improvement: value function shaping
                with tf.variable_scope('vfn'):
                    fitted_value_fn_n = value_fn_arch(self.nobs_t, dout=1)
                with tf.variable_scope('vfn', reuse=True):
                    self.value_fn = fitted_value_fn = value_fn_arch(self.obs_t, dout=1)

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma * fitted_value_fn_n
                log_p_tau = self.reward + self.gamma * fitted_value_fn_n - fitted_value_fn

            log_q_tau = self.lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau - log_pq)
            cent_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))
            self.discriminator_predict = tf.cast(log_p_tau > log_q_tau, tf.float32)
            self.acc = tf.reduce_mean(self.discriminator_predict * self.labels +
                                      (1 - self.discriminator_predict) * (1 - self.labels))

            self.loss = cent_loss + self.reg_loss_skill + self.reg_loss_task
            tot_loss = self.loss
            # self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(tot_loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.gradient_all = self.optimizer.compute_gradients(tot_loss)
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
        obs, obs_next, acts, acts_next, path_probs, reward_weights = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs', 'reward_weights'))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs, expert_reward_weights = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs', 'reward_weights'))

        self.center_reward_gradients = {}
        for var_name, value in zip(self.gradient_task_var_name, self.gradient_task_value):
            self.center_reward_gradients[var_name] = np.zeros(shape=value.shape)

        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch, reward_weights_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, reward_weights, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch, expert_reward_weights_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, expert_reward_weights,
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
            reward_weights_batch = np.concatenate([reward_weights_batch, expert_reward_weights_batch], axis=0)
            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr,
                self.weights: reward_weights_batch
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
                                                                        self.labels: labels, self.lprobs: lprobs_batch,
                                                                        self.weights: reward_weights_batch})
                logger.record_tabular('Discriminator_acc', acc)
                logger.record_tabular('RegLossPeri', mean_reg_loss_peri)
                logger.record_tabular('RegLossCenter', mean_reg_loss_center)

        if logger:
            logger.record_tabular('GCLDiscrimLoss', mean_loss)
            logger.record_tabular('GCLDiscrimRegLossPeri', mean_reg_loss_peri)
            logger.record_tabular('GCLDiscrimRegLossCenter', mean_reg_loss_center)
            # obs_next = np.r_[obs_next, np.expand_dims(obs_next[-1], axis=0)]
            energy, reward_task, reward_skill_1, reward_skill_2, logZ, dtau = \
                tf.get_default_session().run([self.reward, self.reward_task, self.reward_skill_1, self.reward_skill_2,
                                              self.value_fn, self.discrim_output],
                                             feed_dict={self.act_t: acts, self.obs_t: obs, self.nobs_t: obs_next,
                                                        self.lprobs: np.expand_dims(path_probs, axis=1),
                                                        self.weights: reward_weights})
            energy = -energy
            logger.record_tabular('GCLLogZ', np.mean(logZ))
            logger.record_tabular('GCLAverageEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageRewardTask', np.mean(reward_task))
            logger.record_tabular('GCLAverageRewardSkill1', np.mean(reward_skill_1))
            logger.record_tabular('GCLAverageRewardSkill2', np.mean(reward_skill_2))
            logger.record_tabular('GCLAverageLogPtau', np.mean(-energy - logZ))
            logger.record_tabular('GCLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('GCLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('GCLAverageDtau', np.mean(dtau))

            # expert_obs_next = np.r_[expert_obs_next, np.expand_dims(expert_obs_next[-1], axis=0)]
            energy, reward_task, reward_skill_1, reward_skill_2, logZ, dtau = \
                tf.get_default_session().run([self.reward, self.reward_task, self.reward_skill_1, self.reward_skill_2,
                                              self.value_fn, self.discrim_output],
                                             feed_dict={self.act_t: expert_acts,
                                                        self.obs_t: expert_obs,
                                                        self.nobs_t: expert_obs_next,
                                                        self.lprobs: np.expand_dims(expert_probs,
                                                                                    axis=1),
                                                        self.weights: expert_reward_weights})
            energy = -energy
            logger.record_tabular('GCLAverageExpertEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageExpertRewardTask', np.mean(reward_task))
            logger.record_tabular('GCLAverageExpertRewardSkill1', np.mean(reward_skill_1))
            logger.record_tabular('GCLAverageExpertRewardSkill2', np.mean(reward_skill_2))
            logger.record_tabular('GCLAverageExpertLogPtau', np.mean(-energy - logZ))
            logger.record_tabular('GCLAverageExpertLogQtau', np.mean(expert_probs))
            logger.record_tabular('GCLMedianExpertLogQtau', np.median(expert_probs))
            logger.record_tabular('GCLAverageExpertDtau', np.mean(dtau))
        return mean_loss

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        if self.score_discrim:
            self._compute_path_probs(paths, insert=True)
            obs, obs_next, acts, path_probs = self.extract_paths(paths, keys=(
            'observations', 'observations_next', 'actions', 'a_logprobs'))
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(self.discrim_output,
                                                  feed_dict={self.act_t: acts, self.obs_t: obs,
                                                             self.nobs_t: obs_next,
                                                             self.lprobs: path_probs})
            score = np.log(scores) - np.log(1 - scores)
            score = score[:, 0]
        else:
            obs, acts = self.extract_paths(paths)
            reward = tf.get_default_session().run(self.reward,
                                                  feed_dict={self.act_t: acts,
                                                             self.obs_t: obs,
                                                             self.weights: np.repeat([kwargs["reward_weights"]], obs.shape[0], axis=0)})
            score = reward[:, 0]
        return self.unpack(score, paths)

    def eval_skill_reward(self, path, **kwargs):
        obs, acts = self.extract_paths(path)
        reward = tf.get_default_session().run(self.reward_skill_weighted,
                                              feed_dict={self.act_t: acts,
                                                         self.obs_t: obs,
                                                         self.weights: np.repeat([kwargs["reward_weights"]], obs.shape[0], axis=0)})
        score = reward[:, 0]
        return score

    def eval_single(self, target_tensor, obs, acts):
        reward = tf.get_default_session().run(target_tensor,
                                              feed_dict={self.obs_t: obs,
                                                         self.act_t: acts})
        score = reward[:, 0]
        return score

    def debug_eval(self, paths, **kwargs):
        obs, acts = self.extract_paths(paths)
        reward, v = tf.get_default_session().run([self.reward, self.value_fn],
                                                 feed_dict={self.act_t: acts, self.obs_t: obs})
        return {
            'reward': reward,
            'value': v,
        }

    def _make_param_ops(self, vs):
        self._params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        self._params.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="center"))
        assert len(self._params) > 0
        self._assign_plc = [tf.placeholder(tf.float32, shape=param.get_shape(),
                                           name='assign_%s' % param.name.replace('/', '_').replace(':', '_'))
                            for param in self._params]
        self._assign_ops = [tf.assign(self._params[i], self._assign_plc[i]) for i in range(len(self._params))]
