import os
from datetime import datetime
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

    # load expert demonstration data from this repo's data generation
    # experts = load_latest_experts('data/inverted_pendulum', n=5, min_return=-10)

    # load expert demonstration data from core_MSD's data generation
    # experts = load_expert_from_core_MSD("/home/zac/Programming/core_MSD/dataset/old_datasets/InvertedPendulum-v2-nonranked-multistyle.pkl",
    #                                     repeat_each_skill=3,
    #                                     separate_styles=False)

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000,  repeat_each_skill=3,
                                                     separate_styles=True)

    # Hyperparameters
    discriminator_update_step = 10
    n_epochs = 2140

    n_timesteps = 1000
    deterministic = False

    now = datetime.now()
    log_path = f"data/inverted_pendulum_airl_batch/{now.strftime('%d_%m_%Y_%H_%M_%S')}"
    assert not os.path.exists(log_path), "log path already exist! "

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
        entropy_weight=0,  # from AIRL: this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        center_grads=False
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        with rllab_logdir(algo=algo, dirname=log_path+'/demo'):
            algo.train()

        post_probs = [new_likelihood(np.array(irl_model.eval_expert_probs(demonstrations[i], policy)))
                      for i in range(len(demonstrations))]

        # Rollout a trajectory
        ob = env.reset()
        # env.render()
        done = False
        # obs, acts = [], []
        imgs = []
        idx = 0
        for timestep in range(n_timesteps):
            act = policy.get_action(ob)
            # print(act)
            # obs.append(ob)
            act_executed = act[1]["mean"] if deterministic else act[0]
            # acts.append(act_executed)
            ob, rew, done, info = env.step(act_executed)
            imgs.append(env.render('rgb_array'))
        save_video(imgs, os.path.join(f"{log_path}/policy_videos/batch.avi"))

        with rllab_logdir(algo=algo, dirname=log_path + '/likelihood'):
            d = [i for i in range(len(demonstrations))]
            logger.record_tabular(f'Demonstrations', d)
            logger.record_tabular(f'New_Likelihoods', post_probs)
            logger.dump_tabular(with_prefix=False, write_header=True)

        saver = tf.train.Saver(save_dictionary)
        saver.save(sess, f"{log_path}/model.ckpt")


if __name__ == "__main__":
    main()
