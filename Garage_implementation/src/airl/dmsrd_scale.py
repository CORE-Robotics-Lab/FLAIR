import random
from datetime import datetime
import os
import dowel
import numpy as np
from dowel import logger, tabular

from airl.irl_trpo import TRPO
from airl.test_performance import *
from garage.experiment import Snapshotter
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from models.dmsrd_enforce import ReLUModel, AIRLMultiStyleDynamic
from models.fusion_manager import RamFusionDistr
# from scripts.global_utils.utils import *
from garage.experiment import SnapshotConfig


def save_video(ims, filename, fps=30.0):
    import cv2
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()


class MixturePolicy:
    def __init__(self, policies, weights):
        self.policies = policies
        self.weights = np.array(weights)

    def get_action(self, obs):
        return np.dot(self.weights, [policy.get_action(obs)[0] for policy in self.policies]), {}

    def reset(self):
        None

    def get_param_values(self):
        return self.weights.copy()

    def set_param_values(self, weights):
        self.weights = weights


# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_tree(x, xp, k):
    # https://github.com/BiuBiuBiLL/NPEET_LNC/blob/master/lnc.py
    # https://github.com/scipy/scipy/issues/9890 p=2 or np.inf
    tree = cKDTree(x)
    return np.clip(tree.query(xp, k=k + 1, p=float('inf'))[0][:, k], 1e-30, None) # chebyshev distance of k+1-th nearest neighbor


def kldiv(x, xp, k=3):
    """ KL Divergence between p and q for x~p(x), xp~q(x)
    """
    x, xp = np.asarray(x), np.asarray(xp)
    assert k < min(x.shape[0], xp.shape[0]), "Set k smaller than num. samples - 1"
    assert x.shape[1] == xp.shape[1], "Two distributions must have same dim."

    x, xp = x.reshape(x.shape[0], -1), xp.reshape(xp.shape[0], -1)
    n, d = x.shape
    m, _ = xp.shape
    x = add_noise(x)  # fix np.log(0)=inf issue

    const = np.log(m) - np.log(n - 1)
    nn = query_tree(x, x, k)
    nnp = query_tree(xp, x, k - 1)  # (m, k-1)
    return const + d * (np.log(nnp).mean() - np.log(nn).mean())


class DMSRDScale:
    """
    Base Dynamic Multi Style Reward Distillation framework
    """
    def __init__(self, env, env_test, demonstrations, log_prefix, trajs_path, reward_path, bool_save_vid=False,
                 n_workers_irl=10, batch_size=10000, episode_length=1000,
                 batch_when_finding_mixture=3, mix_repeats=1, grid_shots=10000,
                 state_only=True, airl_itrs=600, discriminator_update_step=10, entropy_weight=0.0, metrics_repeat=1,
                 new_strategy_threshold=1.0, msrd_itrs=400, start_msrd=3, l2_reg_strategy=0.01, l2_reg_task=0.0001):
        """
        Initialization

        Args:
            env: train environment
            env_test: test environment

            demonstrations: demonstration list ([strategy, repeat, dictionary])
            log_prefix: log prefix path
            trajs_path: test trajectory set for calculating reward function correlation
            reward_path: ground truth reward information for test trajectory set
            bool_save_vid: whether to save video

            n_workers_irl: how many workers to start in RaySampler for env interaction for AIRL and MSRD
            batch_size: how many environment interactions to sample before training in each iteration
            episode_length: maximum length of an episode

            batch_when_finding_mixture: the batch size when evaluating policy mixture
            mix_repeats: how many repeats of evaluation for each policy mixture weight
            grid_shots: how many different mixture weights to try for policy mixture

            state_only: whether to parameterize the AIRL reward function with only state (for more information, see AIRL paper)
            airl_itrs: how many iterations of AIRL to run for each new strategy (if needed)
            discriminator_update_step: how many discriminator update steps in each iteration
            entropy_weight: entropy weight during AIRL policy training

            new_strategy_threshold: the threshold to skip AIRL training and directly take the policy mixture solution
            msrd_itrs: how many iterations of MSRD to run
            start_msrd: start MSRD training after which DMSRD iteration (for example, the third iteration/demonstration available to DMSRD)
            l2_reg_strategy: MSRD strategy regularization
            l2_reg_task: MSRD task regularization
        """
        # envs
        self.env = env
        self.env_test = env_test

        # loading and saving
        self.demonstrations = demonstrations
        now = datetime.now()
        self.log_path = f"data/{log_prefix}/{now.strftime('%Y_%m_%d_%H_%M_%S')}"
        assert not os.path.exists(self.log_path), "log path already exist! "
        os.mkdir(self.log_path)
        self.trajs_path = trajs_path
        self.reward_path = reward_path
        self.bool_save_vid = bool_save_vid

        # sampler and environment interaction parameters
        self.n_workers_irl = n_workers_irl
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.metrics_repeat = metrics_repeat

        # policy mixture optimization parameters
        assert batch_when_finding_mixture % mix_repeats == 0, "each batch should contain full repeats of weights"
        self.batch_when_finding_mixture = batch_when_finding_mixture
        self.mix_repeats = mix_repeats
        self.grid_shots = grid_shots

        # airl parameters
        self.state_only = state_only
        self.airl_itrs = airl_itrs
        self.discriminator_update_step = discriminator_update_step
        self.entropy_weight = entropy_weight

        # msrd and dmsrd parameters
        self.new_strategy_threshold = new_strategy_threshold
        self.msrd_itrs = msrd_itrs
        self.start_msrd = start_msrd
        self.l2_reg_strategy = l2_reg_strategy,
        self.l2_reg_task = l2_reg_task,

        # initialize dmsrd components
        self.task_reward = self._get_reward_model(scope="task")
        self.mixture_weights = []  # for each demonstration
        self.policies = []
        self.fusions = []  # experience-replay-like buffer for AIRL discriminator training
        self.strategy_rewards = []
        self.strategy_reward_vars = []  # identify which vars to update during training
        self.reward_fs = []  # combination of task reward and strategy reward
        self.value_fs = []
        self.value_vars = []  # identify which vars to update during training
        self.baselines = []  # baseline value dependent on the state TODO: check
        self.trainers = []
        self.algos = []
        self.to_remove = []

        self.experts_multi_styles = []  # demonstrations available at each iteration
        self.strategy_to_demo = {}  # map from strategy to the pure demonstration

        self.save_dictionary = {}  # the dictionary of variables to save
        self.new_dictionary = {}  # the dictionary of new variables introduced in each iteration by AIRL
        self.task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task')  # identify which vars to update during training
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='task')):
            self.save_dictionary[f'my_task_{idx}'] = var

        # some state variables
        self.num_strategies = 0
        self.new_pol = True  # state variable indicating whether the previous iteration chose the new AIRL policy

        # random order of the demonstrations being available
        self.rand_idx = random.sample(range(len(self.demonstrations)), len(self.demonstrations))

    def _get_reward_model(self, scope):
        return ReLUModel(scope, self.env.observation_space.shape[0]) \
            if self.state_only \
            else ReLUModel(scope, self.env.observation_space.shape[0] + self.env.action_space.shape[0])

    @staticmethod
    def _create_tf_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    @staticmethod
    def _switch_to_new_logger_dir(path):
        # reset logger to the new strategy log dir
        logger.remove_all()
        logger.add_output(dowel.CsvOutput(path))
        logger.add_output(dowel.StdOutput())

    def _run_policy_get_observation(self, n_workers, sampler=None, shutdown=True, policy=None, policy_update=None):
        assert not (sampler is None and policy is None), "Either sampler or policy needs to be available!"
        if sampler is None:
            sampler = RaySampler(agents=policy, envs=self.env,
                                 max_episode_length=self.env.spec.max_episode_length,
                                 is_tf_worker=True) #, n_workers=n_workers)
        sampler.start_worker()
        observations = [ep["observations"] for ep in
                        sampler.obtain_exact_episodes(n_eps_per_worker=1, agent_update=policy_update).to_list()]
        if shutdown:
            sampler.shutdown_worker()
        return observations

    def new_demonstration(self, iteration):
        """
        Create reward and policies computation graph for new demonstration
        """
        idx = self.rand_idx.pop(0)
        self.experts_multi_styles.append(self.demonstrations[idx])

        if self.new_pol:
            # last AIRL training was accepted, and therefore we need to create new computation graph for a new strategy
            # otherwise we could just use the previous computation graph
            new_strategy_ind = len(self.strategy_rewards)

            # garage trainer for logging and saving (not using TFTrainer as we are handling the tf session on our own)
            trainer = Trainer(SnapshotConfig(snapshot_dir=f'{self.log_path}/strategy_{new_strategy_ind}',
                                             snapshot_mode='last',
                                             snapshot_gap=1))

            new_strategy_reward_scope = f"strategy_{new_strategy_ind}"
            new_strategy_value_scope = f"value_{new_strategy_ind}"
            new_strategy_reward = self._get_reward_model(new_strategy_reward_scope)
            value_fn = self._get_reward_model(new_strategy_value_scope)

            # update all components of DMSRD

            self.strategy_reward_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=new_strategy_reward_scope))
            self.strategy_rewards.append(new_strategy_reward)
            self.value_vars.append(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=new_strategy_value_scope))
            self.value_fs.append(value_fn)

            self.new_dictionary = {}
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=new_strategy_reward_scope)):
                self.new_dictionary[f'my_strategy_{new_strategy_ind}_{idx}'] = var
            for idx, var in enumerate(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=new_strategy_value_scope)):
                self.new_dictionary[f'my_value_{new_strategy_ind}_{idx}'] = var

            self.baselines.append(LinearFeatureBaseline(env_spec=self.env.spec))
            self.fusions.append(RamFusionDistr(10000, subsample_ratio=1))
            self.trainers.append(trainer)

        # Each of the previous demonstration has zero weight on the new strategy
        [c_i.append(0.0) for c_i in self.mixture_weights]
        # Last demonstration is purely on the new strategy
        self.mixture_weights.append([0.0] * len(self.strategy_rewards))
        self.mixture_weights[-1][-1] = 1.0

        # pass the mixture weights into trajectories
        # for idx, style in enumerate(self.experts_multi_styles):
        #     for path in style:
        #         path["reward_weights"] = np.repeat([self.mixture_weights[idx]], len(path["observations"]), axis=0)

    def _train_airl(self, policy, trainer):
        self._switch_to_new_logger_dir(os.path.join(trainer._snapshotter.snapshot_dir, 'airl_progress.csv'))
        sampler = RaySampler(agents=policy, envs=self.env,
                             max_episode_length=self.env.spec.max_episode_length,
                             is_tf_worker=True, n_workers=self.n_workers_irl)
        trainer._sampler = sampler
        trainer._start_worker()
        trainer.train(n_epochs=self.airl_itrs, batch_size=self.batch_size)
        trainer._shutdown_worker()

    def mixture_optimize(self, iteration):
        """
        Perform mixture optimization and determine if new strategy needs to be created for demonstration
        """
        self.num_strategies = len(self.strategy_rewards)  # including the new un-trained strategy

        with self._create_tf_session() as sess:
            self.policies = [GaussianMLPPolicy(name=f'policy_{strategy}',
                                               env_spec=self.env.spec,
                                               hidden_sizes=[32, 32]) for strategy in range(self.num_strategies)]

            # build AIRL computation graph for the new strategy
            strategy = self.num_strategies - 1
            for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{strategy}')):
                self.new_dictionary[f'my_policy_{strategy}_{idx}'] = var
            with tf.variable_scope(f"iter_{iteration}_new_strategy"):
                irl_var_list = self.strategy_reward_vars[strategy] + self.value_vars[strategy]
                irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                                  self.strategy_rewards[strategy],
                                                  self.value_fs[strategy],
                                                  expert_trajs=self.experts_multi_styles[-1],
                                                  var_list=irl_var_list,
                                                  state_only=self.state_only,
                                                  fusion=self.fusions[strategy],
                                                  new_strategy=True,
                                                  l2_reg_skill=self.l2_reg_strategy,
                                                  l2_reg_task=self.l2_reg_task,
                                                  max_itrs=self.discriminator_update_step)

                algo = TRPO(env_spec=self.env.spec,
                            policy=self.policies[strategy],
                            baseline=self.baselines[strategy],
                            index=strategy,
                            center_grads=False,
                            sampler=None,
                            irl_model=irl_model,
                            generator_train_itrs=1,
                            discrim_train_itrs=10,
                            discount=0.99,
                            max_kl_step=0.01)

                self.trainers[strategy].setup(algo, self.env)

            sess.run(tf.global_variables_initializer())
            if iteration > 0:
                # restore training from previous iterations
                saver = tf.train.Saver(self.save_dictionary)
                saver.restore(sess, f"{self.log_path}/model.ckpt")

            # if self.bool_save_vid:
            #     self.save_video(iteration)

            policies_for_mixture = np.array(self.policies[:strategy])
            num_pols = len(policies_for_mixture)
            new_demo = self.experts_multi_styles[-1]
            if num_pols > 0:
                # mixture optimization

                # observation kl estimation preparation (for the new demonstration)
                assert np.all([demo["observations"].shape[0] == self.episode_length] for demo in new_demo), \
                    "the expert episode length needs to equal max episode length"
                expert_observations = np.concatenate([demo["observations"] for demo in new_demo])
                # remaining_demos = [np.concatenate(
                #     [demo["observations"] for demo in self.demonstrations[rem_idx]]) for rem_idx in self.rand_idx]

                mixture_samples = 0
                if num_pols > 1:
                    # need to do mixture optimization
                    # generate random weights
                    weights = np.eye(num_pols)
                    weights = np.append(weights, np.random.uniform(0, 1, (self.grid_shots, num_pols)), axis=0)
                    # make sure all weights are legal
                    for i in range(len(weights)):
                        assert np.all(weights[i]) >= 0
                        weights[i] /= np.sum(weights[i], dtype=np.float64)
                    # for each weight, we repeat self.mix_repeats times to evaluate its goodness
                    # weight_unique = weights
                    weights = np.repeat(weights, self.mix_repeats, axis=0)
                    weights_len = weights.shape[0]

                    # div_matrix = np.ones((len(self.rand_idx), len(weight_unique))) * 1e10

                    rounds = weights_len // self.batch_when_finding_mixture
                    best_mixture_distance = 1e10
                    best_mixture_weight = None

                    # take the first batch of weight to initialize the RaySampler
                    mixture_policies = [MixturePolicy(policies_for_mixture, weight)
                                        for weight in weights[0:self.batch_when_finding_mixture]]
                    sampler = RaySampler(agents=mixture_policies, envs=self.env,
                                         max_episode_length=self.env.spec.max_episode_length,
                                         is_tf_worker=True) #, n_workers=self.batch_when_finding_mixture)
                    for i in range(rounds):
                        weights_this_round = weights[i*self.batch_when_finding_mixture:(i+1)*self.batch_when_finding_mixture]
                        policy_updates = []
                        for idx, mixture_policy in enumerate(mixture_policies):
                            mixture_policy.weights = weights_this_round[idx]
                            policy_updates.append(mixture_policy.get_param_values())
                        observations_mixture = self._run_policy_get_observation(policy_update=policy_updates,
                                                                                n_workers=self.batch_when_finding_mixture,
                                                                                sampler=sampler,
                                                                                shutdown=False)
                        mixture_samples += self.batch_when_finding_mixture

                        # calculate the kl-divergence for each weight
                        kl_divergence_this_round = \
                            [kldiv(expert_observations,
                                   np.concatenate(
                                       observations_mixture[idx * self.mix_repeats:(idx + 1) * self.mix_repeats]),
                                   3) for idx in range(self.batch_when_finding_mixture // self.mix_repeats)]

                        # for demo_idx, rem_demo in enumerate(remaining_demos):
                        #     div_matrix[demo_idx][i*rounds*self.batch_when_finding_mixture//self.mix_repeats:(i+1)*rounds*self.batch_when_finding_mixture//self.mix_repeats] = [kldiv(rem_demo,
                        #            np.concatenate(
                        #                observations_mixture[idx * self.mix_repeats:(idx + 1) * self.mix_repeats]),
                        #            3) for idx in range(self.batch_when_finding_mixture // self.mix_repeats)]

                        best_idx_this_round = np.argmin(kl_divergence_this_round)
                        if kl_divergence_this_round[best_idx_this_round] < best_mixture_distance:
                            best_mixture_distance = kl_divergence_this_round[best_idx_this_round]
                            best_mixture_weight = weights_this_round[best_idx_this_round * self.mix_repeats]
                            # if best_mixture_distance < self.new_strategy_threshold:
                            #     break
                    sampler.shutdown_worker()
                    # Remaining demos find mixtures
                    # min_divs = np.min(div_matrix, axis=1)
                    # argmin_divs = np.argmin(div_matrix, axis=1)
                    # indices_mins = np.argwhere(
                    #     min_divs < self.new_strategy_threshold)
                    # self.metrics_repeat = len(indices_mins) + 1
                    # mix_weights = weight_unique[argmin_divs[indices_mins]]
                    # self.to_remove = []
                    # for i in range(len(indices_mins)):
                    #     rem_demo = self.demonstrations[
                    #         self.rand_idx[indices_mins[i,0]]]
                    #     self.experts_multi_styles.insert(
                    #         len(self.experts_multi_styles) - 1, rem_demo)
                    #     self.mixture_weights.insert(
                    #         len(self.mixture_weights) - 1,
                    #         [0.0] * len(self.strategy_rewards))
                    #     new_list = mix_weights[i].tolist()[0]
                    #     new_list.extend([0.0])
                    #     self.mixture_weights[-2] = new_list
                    #     self.to_remove.append(indices_mins[i,0])
                    # for j_idx in reversed(self.to_remove):
                    #     self.rand_idx.pop(j_idx)
                else:
                    # if num_pols == 1, no need to do mixture optimization, just use the single policy available
                    policy = self.policies[0]
                    observations_mixture = self._run_policy_get_observation(policy=policy, n_workers=self.mix_repeats)
                    observations_mixture = np.concatenate(observations_mixture, axis=0)
                    best_mixture_distance = kldiv(expert_observations, observations_mixture, 3)
                    best_mixture_weight = np.array([1.0])

                new_airl_distance = 1e10
                if best_mixture_distance > self.new_strategy_threshold:
                    # try train new strategy
                    strategy = self.num_strategies - 1
                    self._train_airl(self.policies[strategy], self.trainers[strategy])
                    observations_mixture = self._run_policy_get_observation(policy=self.policies[strategy], n_workers=self.mix_repeats)
                    observations_mixture = np.concatenate(observations_mixture, axis=0)
                    new_airl_distance = kldiv(expert_observations, observations_mixture, 3)

                self._switch_to_new_logger_dir(os.path.join(f'{self.log_path}/mixture', 'progress.csv'))
                tabular.record(f'Demonstration', self.rand_idx[iteration])
                tabular.record(f'Best_Mixture', str(best_mixture_weight))
                tabular.record(f'Mixture_distance', best_mixture_distance)
                tabular.record(f'New_airl_distance', new_airl_distance)
                tabular.record(f'Samples', mixture_samples)
                logger.log(tabular)
                logger.dump_all(iteration)
                tabular.clear()

                ratio = new_airl_distance < best_mixture_distance
                if not ratio or best_mixture_distance < self.new_strategy_threshold:
                    # accept mixture
                    self.new_pol = False
                    [c_i.pop() for c_i in self.mixture_weights]
                    self.mixture_weights[-1] = (best_mixture_weight / np.sum(best_mixture_weight)).tolist()
                else:
                    # accept AIRL new policy
                    self.strategy_to_demo[strategy] = iteration
                    self.save_dictionary.update(self.new_dictionary)
                    self.new_pol = True
                # if self.bool_save_vid:
                #     self.save_mixture_video(self.policies[:-1], iteration)
            else:
                # no policy available yet to do policy mixture
                strategy = self.num_strategies - 1
                self._train_airl(self.policies[strategy], self.trainers[strategy])
                self.strategy_to_demo[strategy] = iteration  # update pure strategy
                self.save_dictionary.update(self.new_dictionary)  # accept AIRL results into DMSRD
                self.new_pol = True

            saver = tf.train.Saver(self.save_dictionary)
            saver.save(sess, f"{self.log_path}/model.ckpt")

    def msrd(self, iteration, no_train=False):
        # start msrd training after self.start_msrd to avoid biasing the task reward
        self.num_strategies = len(self.strategy_rewards) - 1
        if self.new_pol:
            # decides to include the newly trained AIRL
            self.num_strategies += 1
        self.reward_fs = []
        self.algos = []
        self.policies = []
        with self._create_tf_session() as sess:
            for strategy in range(self.num_strategies):
                policy = GaussianMLPPolicy(name=f'policy_{strategy}',
                                           env_spec=self.env.spec,
                                           hidden_sizes=(32, 32))
                self.policies.append(policy)
                with tf.variable_scope(f"iter_{iteration}_strategy_{strategy}"):
                    irl_var_list = self.task_vars + self.strategy_reward_vars[strategy] + self.value_vars[strategy]
                    irl_model = AIRLMultiStyleDynamic(self.env, self.task_reward,
                                                      self.strategy_rewards[strategy],
                                                      self.value_fs[strategy],
                                                      skill_value_var_list=self.strategy_reward_vars[strategy],
                                                      expert_trajs=self.experts_multi_styles[self.strategy_to_demo[strategy]],
                                                      mix_trajs=np.array(self.experts_multi_styles)[np.arange(len(self.experts_multi_styles)) != self.strategy_to_demo[strategy]],
                                                      reward_weights=np.array(self.mixture_weights)[:, strategy][np.arange(len(self.experts_multi_styles)) != self.strategy_to_demo[strategy]],
                                                      var_list=irl_var_list,
                                                      state_only=self.state_only,
                                                      new_strategy=False,
                                                      fusion=self.fusions[strategy],
                                                      l2_reg_skill=self.l2_reg_strategy,
                                                      l2_reg_task=self.l2_reg_task,
                                                      max_itrs=self.discriminator_update_step)

                    reward_weights = [0.0] * self.num_strategies
                    reward_weights[strategy] = 1.0
                    algo = TRPO(env_spec=self.env.spec,
                                policy=self.policies[strategy],
                                baseline=self.baselines[strategy],
                                index=strategy,
                                center_grads=True,
                                sampler=None,
                                irl_model=irl_model,
                                generator_train_itrs=1,
                                discrim_train_itrs=10,
                                discount=0.99,
                                max_kl_step=0.01)
                    self.trainers[strategy].setup(algo, self.env)
                    self.reward_fs.append(irl_model)
                    self.algos.append(algo)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.save_dictionary)
            saver.restore(sess, f"{self.log_path}/model.ckpt")

            if self.new_pol and not no_train:
                # MSRD
                for strategy in range(self.num_strategies):
                    sampler = RaySampler(agents=self.policies[strategy],
                                         envs=self.env,
                                         max_episode_length=self.env.spec.max_episode_length,
                                         is_tf_worker=True, n_workers=self.n_workers_irl)
                    self.trainers[strategy]._sampler = sampler
                    self.trainers[strategy]._start_worker()
                for epoch in range(self.msrd_itrs):
                    center_reward_gradients = None
                    for strategy in range(self.num_strategies):
                        trainer = self.trainers[strategy]
                        self._switch_to_new_logger_dir(os.path.join(trainer._snapshotter.snapshot_dir, 'msrd_progress.csv'))
                        trainer.train(n_epochs=epoch + 1, batch_size=self.batch_size,
                                      start_epoch=epoch)
                        if center_reward_gradients is None:
                            center_reward_gradients = self.algos[strategy].center_reward_gradients
                        else:
                            assert center_reward_gradients.keys() == self.algos[strategy].center_reward_gradients.keys()
                            for key in center_reward_gradients.keys():
                                center_reward_gradients[key] += self.algos[strategy].center_reward_gradients[key]
                    feed_dict = {}
                    assert self.task_reward.grad_map_vars.keys() == center_reward_gradients.keys()
                    for key in self.task_reward.grad_map_vars.keys():
                        feed_dict[self.task_reward.grad_map_vars[key]] = center_reward_gradients[key]
                    sess.run(self.task_reward.step, feed_dict=feed_dict)
                for strategy in range(self.num_strategies):
                    self.trainers[strategy]._shutdown_worker()

                saver = tf.train.Saver(self.save_dictionary)
                saver.save(sess, f"{self.log_path}/model.ckpt")

            # load test trajectory set
            trajectories = []
            for i in range(40):
                with open(f'data/{self.trajs_path}/trajectories_{i}.pkl', "rb") as f:
                    trajectories.extend(pickle.load(f))

            # load test trajectory ground-truth reward
            ground_truths = []
            with open(f'data/{self.reward_path}.csv',
                      newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ',
                                    quotechar='|')
                first = True
                for row in reader:
                    if first:
                        first = False
                        continue
                    ground_truths.append(float(row[0]))
            rews, divs, likes, grounds, abs = [], [], [], [], []
            for i in range(self.metrics_repeat):
                likelhi = get_dmsrd_likelihood(self.experts_multi_styles,
                                               self.policies,
                                               self.reward_fs[0],
                                               self.mixture_weights)
                likes.append(np.average(likelhi))

                # divergence
                print("calculating reward and divergence...")
                rew, div = get_dmsrd_divergence(self.env_test,
                                                self.mixture_weights,
                                                self.policies,
                                                self.experts_multi_styles,
                                                self.episode_length)
                rews.append(np.average(rew))
                divs.append(np.average(div))

                print("calculating task reward...")
                record = np.zeros(len(trajectories))
                for tidx, traj in enumerate(trajectories):
                    reward_cent = tf.get_default_session().run(
                        self.reward_fs[0].reward_task,
                        feed_dict={self.reward_fs[0].obs_t: traj["observations"]})
                    score = reward_cent[:, 0]
                    record[tidx] = np.mean(score)

                grounds.append(np.corrcoef(ground_truths, record)[0,1])

                rew = []
                for demo in self.experts_multi_styles:
                    strat_rew = []
                    for strat in range(len(self.reward_fs)):
                        rew_repeat = 0
                        for traj in demo:
                            reward = tf.get_default_session().run(
                                self.reward_fs[strat].reward_skill,
                                feed_dict={
                                    self.reward_fs[strat].obs_t: traj["observations"]})
                            score = np.mean(reward[:, 0])
                            rew_repeat += np.mean(score / len(demo))
                        strat_rew.append(rew_repeat)
                    rew.append(strat_rew)

                rew = np.array(rew)
                for j in range(len(rew[0])):
                    max_reward = np.amax(rew[:, j])
                    rew[:, j] = np.exp(rew[:, j] - max_reward)

                np_mixture_weights = np.array(self.mixture_weights)
                distances = []
                for j in range(len(self.reward_fs)):
                    a = np_mixture_weights[:, j]
                    b = rew[:, j]
                    similiarity = np.dot(a, b.T) / (
                            np.linalg.norm(a) * np.linalg.norm(b))
                    dist = 1. - similiarity
                    distances.append(dist)
                abs.append(np.mean(distances))

            return rews, divs, likes, grounds, abs
