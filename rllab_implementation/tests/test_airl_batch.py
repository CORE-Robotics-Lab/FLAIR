import os
from datetime import datetime
import csv
import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.likelihood_utils import new_likelihood
from global_utils.utils import *
import rllab.misc.logger as logger
import itertools


def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000, repeat_each_skill=3,
                                                     separate_styles=True)

    log_path = "data/inverted_pendulum_airl_batch/16_03_2022_21_14_34"

    # Hyperparameters
    discriminator_update_step = 10
    n_epochs = 1

    n_timesteps = 1000
    deterministic = False

    save_dictionary = {}
    irl_model = AIRL(env=env, expert_trajs=list(itertools.chain(*demonstrations)), state_only=True, fusion=True, max_itrs=discriminator_update_step, name='reward_0')
    for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reward_0')):
        save_dictionary[f'my_skill_{0}_{idx}'] = var

    policy = GaussianMLPPolicy(name='policy_0', env_spec=env.spec, hidden_sizes=(32, 32))

    for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_0')):
        save_dictionary[f'my_policy_{0}_{idx}'] = var
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=n_epochs,
        batch_size=10000,
        max_path_length=1000,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=discriminator_update_step,
        irl_model_wt=1.0,
        entropy_weight=0.1,  # from AIRL: this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model.ckpt")

        with rllab_logdir(algo=algo, dirname=log_path + '/reward'):
            task_rewards = []
            for traj in trajectories:
                reward = tf.get_default_session().run(irl_model.reward,
                                                      feed_dict={irl_model.obs_t: traj})
                score = reward[:, 0]
                task_rewards.append(np.mean(score))
                logger.record_tabular(f'AIRL Policy', np.mean(score))
                logger.dump_tabular(with_prefix=False, write_header=True)
            logger.record_tabular(f'Correlation', np.corrcoef(ground_truths, task_rewards))
            logger.dump_tabular(with_prefix=False, write_header=True)

        post_likelihoods = [new_likelihood(np.array(irl_model.eval_expert_probs(demonstrations[i], policy, fix=True)))
                      for i in range(len(demonstrations))]
        post_probs = [new_likelihood(np.array(irl_model.eval_numerical_integral(demonstrations[i], policy)))
                      for i in range(len(demonstrations))]

        with rllab_logdir(algo=algo, dirname=log_path + '/probs'):
            logger.record_tabular(f'Final_Likelihoods', post_likelihoods)
            logger.record_tabular(f'Final_Probs', post_probs)
            logger.dump_tabular(with_prefix=False, write_header=True)

        print(post_likelihoods)
        print(post_probs)


if __name__ == "__main__":
    main()
