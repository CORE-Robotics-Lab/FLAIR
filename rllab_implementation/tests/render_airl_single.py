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
from inverse_rl.models.dmsrd import new_likelihood
from global_utils.utils import *
import rllab.misc.logger as logger
# from tests.heatmap import heatmap, annotate_heatmap
# import matplotlib.pyplot as plt


def main():
    # env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    env = TfEnv(CustomGymEnv('InvertedDoublePendulum-v2', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    # demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000, repeat_each_skill=3,
    #                                                  separate_styles=True)

    demonstrations = load_expert_from_core_MSD('data/InvertedDoublePendulum10skills.pkl', length=1000, repeat_each_skill=3,
                                                     separate_styles=True)

    # Hyperparameters
    discriminator_update_step = 10
    n_epochs = 1000

    n_timesteps = 1000
    deterministic = False

    now = datetime.now()
    log_path = f"data/inverted_pendulum_airl_single/19_10_2021_13_31_46"

    irl_models = []
    policies = []
    algos = []

    save_dictionary = {}
    for index in range(4):
        irl_model = AIRL(env=env, expert_trajs=demonstrations[index], state_only=True, fusion=True,
                         max_itrs=discriminator_update_step, name=f'skill_{index}')
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{index}')):
            save_dictionary[f'my_skill_{index}_{idx}'] = var

        policy = GaussianMLPPolicy(name=f'policy_{index}', env_spec=env.spec, #std_share_network=True,
                                   hidden_sizes=(32, 32))
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{index}')):
            save_dictionary[f'my_policy_{index}_{idx}'] = var

        algo = IRLTRPO(
            env=env,
            policy=policy,
            irl_model=irl_model,
            n_itr=n_epochs,
            batch_size=10000,
            max_path_length=1000,
            discount=0.99,
            store_paths=False,
            discrim_train_itrs=discriminator_update_step,
            irl_model_wt=1.0,
            entropy_weight=0.1,  # from AIRL: this should be 1.0 but 0.1 seems to work better
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=env.spec)
        )
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model.ckpt")

        for skill in range(len(policies)):
            # Rollout a trajectory
            ob = env.reset()
            # env.render()
            done = False
            # obs, acts = [], []
            imgs = []
            idx = 0
            policy = policies[skill]
            for timestep in range(n_timesteps):
                act = policy.get_action(ob)
                # print(act)
                # obs.append(ob)
                act_executed = act[1]["mean"] if deterministic else act[0]
                # acts.append(act_executed)
                ob, rew, done, info = env.step(act_executed)
                imgs.append(env.render('rgb_array'))
            save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{skill}.avi"))

        saver = tf.train.Saver(save_dictionary)
        saver.save(sess, f"{log_path}/model.ckpt")


if __name__ == "__main__":
    main()
