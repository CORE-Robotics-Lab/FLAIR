import argparse
import pickle
import tensorflow as tf
from datetime import datetime

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from global_utils.utils import *
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.airl_state import AIRL
from inverse_rl.utils.log_utils import rllab_logdir
import itertools


def main():
    env = TfEnv(CustomGymEnv('Ant-v2', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/Ant10skills.pkl', length=1000, repeat_each_skill=3,
                                                     separate_styles=True)

    # Hyperparameters
    discriminator_update_step = 10
    n_epochs = 2000
    now = datetime.now()
    log_path = f"data/ant_airl_batch/{now.strftime('%d_%m_%Y_%H_%M_%S')}"

    irl_model = AIRL(env=env, expert_trajs=list(itertools.chain(*demonstrations)), state_only=True, fusion=True, max_itrs=discriminator_update_step,
                     score_discrim=False)

    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

    sampler_class = VectorizedSampler
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
        sampler_cls=sampler_class,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        gae_lambda=0.97,
        step_size=0.01,
        optimizer_args=dict(reg_coeff=0.1, cg_iters=10),
        reward_batch_size=512,
        num_policy_steps=3,
    )

    with rllab_logdir(algo=algo, dirname=log_path):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            algo.train()


if __name__ == "__main__":
    main()