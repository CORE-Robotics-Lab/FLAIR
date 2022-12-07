import os
import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *


def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstration data from this repo's data generation
    # experts = load_latest_experts('data/inverted_pendulum', n=5, min_return=-10)

    # load expert demonstration data from core_MSD's data generation
    experts = load_expert_from_core_MSD("/home/zac/Programming/core_MSD/dataset/old_datasets/InvertedPendulum-v2-nonranked-multistyle.pkl",
                                        repeat_each_skill=3,
                                        separate_styles=False)

    # Hyperparameters
    discriminator_update_step = 10
    n_epochs = 1000
    log_path = "data/inverted_pendulum_styled_airl_4"

    # irl_model = AIRLStateAction(env_spec=env.spec, expert_trajs=experts)
    irl_model = AIRL(env=env, expert_trajs=experts, state_only=True, fusion=True, max_itrs=discriminator_update_step)

    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
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

    assert not os.path.exists(log_path), "log path already exist! "
    with rllab_logdir(algo=algo, dirname=log_path):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            algo.train()


if __name__ == "__main__":
    main()
