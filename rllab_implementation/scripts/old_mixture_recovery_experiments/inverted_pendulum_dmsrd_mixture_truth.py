import os

import random
from datetime import datetime
import itertools

import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.dmsrd import ReLUModel, AIRLMultiStyleDynamic, findMixture, new_likelihood
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger

def safe_log_np(x):
    x = np.clip(x, 1e-37, 100)
    return np.log(x)

def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    # demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skills.pkl', length=100, repeat_each_skill=3,
    #                                                  separate_styles=True)

    skills = [0, 3, 8]
    # skills = [3, 5, 7]
    skill_str = "".join([str(num) for num in skills])
    #mixture = [0.2, 0.1, 0.7]
    #mixture = [0.6, 0.1, 0.3]
    mixture = [0.1, 0.8, 0.1]
    mixture_str = "_".join([str(num) for num in mixture])

    num_skills = 3


    # load mixture
    mix = load_expert_from_core_MSD("data/"+skill_str+"_"+mixture_str+"_"+"expertmix.pkl", length=1000, repeat_each_skill=3,
                                                     separate_styles=True)

    # Hyperparameters
    state_only = True
    deterministic = False
    n_timesteps = 1000
    load_path = f"data/inverted_pendulum_dmsrd_mixture/{skill_str}_{mixture_str}expertmix/28_04_2021_21_36_12"
    # load_path = f"data/inverted_pendulum_dmsrd_permute/14_04_2021_17_15_22"
    assert os.path.exists(load_path), "load path does not exist! "

    task_reward = ReLUModel("task", env.observation_space.shape[0]) \
        if state_only \
        else ReLUModel("task", env.observation_space.shape[0] + env.action_space.shape[0])


    strategy_rewards = []

    policies = []
    iteration = 0
    save_dictionary = {}
    for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='task')):
        save_dictionary[f'my_task_{idx}'] = var

    while iteration < num_skills:
        new_skill_reward = ReLUModel(f"skill_{len(strategy_rewards)}", env.observation_space.shape[0]) \
            if state_only \
            else ReLUModel(f"skill_{len(strategy_rewards)}", env.observation_space.shape[0] + env.action_space.shape[0])

        skill_vars = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'skill_{len(strategy_rewards)}'))
        policy = GaussianMLPPolicy(name=f'policy_{len(strategy_rewards)}', env_spec=env.spec, hidden_sizes=(32, 32))

        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{len(strategy_rewards)}')):
            save_dictionary[f'my_skill_{len(strategy_rewards)}_{idx}'] = var
        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{len(strategy_rewards)}')):
            save_dictionary[f'my_policy_{len(strategy_rewards)}_{idx}'] = var

        policies.append(policy)
        strategy_rewards.append(new_skill_reward)
        iteration+=1

    task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task')
    with tf.variable_scope(f"iter_{iteration}"):
        irl_var_list = task_vars + skill_vars
        irl_model = AIRLMultiStyleDynamic(env, task_reward,
                                          strategy_rewards,
                                          expert_trajs=mix[0],
                                          var_list=irl_var_list,
                                          state_only=state_only,
                                          fusion=True,
                                          l2_reg_skill=0.001,
                                          l2_reg_task=0.0001,
                                          max_itrs=10)

    # Build MSRD computation graph
    num_skills = len(strategy_rewards)
    # Run MSRD
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if iteration > 0:
            saver = tf.train.Saver(save_dictionary)
            saver.restore(sess, f"{load_path}/model.ckpt")

        action_prob = np.array(
            [irl_model.eval_expert_probs(mix[0], policy) for policy in policies], dtype=np.float64)

        print([new_likelihood(prob) for prob in action_prob])

        action_prob = np.clip(action_prob, None, 0.)
        action_prob = np.exp(action_prob)
        action_prob = np.resize(action_prob, (action_prob.shape[0], action_prob.shape[1] * action_prob.shape[2]))

        weights = np.array([[0.1, 0.8, 0.1]], dtype=np.float64)
        out = np.matmul(weights, action_prob)
        out = np.sum(safe_log_np(out), axis=1)
        print(out)


""" Save video of demonstration """
if __name__ == "__main__":
    main()
