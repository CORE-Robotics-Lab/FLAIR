#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
from datetime import datetime
import itertools
import gym
import tensorflow as tf
from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from airl.irl_trpo import TRPO
from models.airl_state import AIRL

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
from airl.test_performance import *

now = datetime.now()
log_path = f"data/lunar_lander_airl_batch/17_01_2022_19_22_01"

irl_models = []
policies = []
algos = []
trainers = []

demonstrations = load_expert_from_core_MSD(
    'data/LunarLander10skills500.pkl', length=500,
    repeat_each_skill=3,
    separate_styles=True)
env = GymEnv('LunarLanderContinuous-v2')


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}
    for index in range(1):
        irl_model = AIRL(env=env, expert_trajs=list(itertools.chain(*demonstrations)),
                         state_only=True, fusion=False,
                         max_itrs=10,
                         name=f'skill_{index}')
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'skill_{index}')):
            save_dictionary[f'my_skill_{index}_{idx}'] = var

        policy = GaussianMLPPolicy(name=f'policy_{index}',
                                   env_spec=env.spec,
                                   hidden_sizes=(32, 32))
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'policy_{index}')):
            save_dictionary[f'my_policy_{index}_{idx}'] = var

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    index=index,
                    sampler=sampler,
                    irl_model=irl_model,
                    generator_train_itrs=1,
                    discrim_train_itrs=10,
                    discount=0.99,
                    max_kl_step=0.01)
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    saver = tf.train.Saver(save_dictionary)
    saver.restore(sess, f"{log_path}/model.ckpt")
    env_test = gym.make('LunarLanderContinuous-v2')
    # for i in range(1):
    #     ob = env_test.reset()
    #     policy = policies[i]
    #     imgs = []
    #     for timestep in range(500):
    #         ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
    #         imgs.append(env_test.render('rgb_array'))
    #     save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{i}.avi"))

    # trajectories = []
    # for i in range(40):
    #     with open(f'data/lunar_trajs/trajectories_{i}.pkl', "rb") as f:
    #         trajectories.extend(pickle.load(f))
    #
    # record = np.zeros(len(trajectories))
    # for tidx, traj in enumerate(trajectories):
    #     reward = tf.get_default_session().run(irl_model.reward,
    #                                           feed_dict={irl_model.obs_t: traj["observations"]})
    #     score = reward[:, 0]
    #     record[tidx] = np.mean(score)

    # with open(f'{log_path}/reward.csv', 'w') as csvfile:
    #     # creating a csv writer object
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["Task Reward"])
    #     csvwriter.writerow(record.tolist())

    likes = get_batch_likelihoods(irl_model, demonstrations, policy)
    rew, div = get_batch_rewdiv(env_test, policy, demonstrations, 500)
    with open(f'{log_path}/probs.csv', 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Likelihoods"])
        csvwriter.writerow(likes)
        csvwriter.writerow(["Reward"])
        csvwriter.writerow(rew)
        csvwriter.writerow(["Divergence"])
        csvwriter.writerow(div)

    # saver = tf.train.Saver(save_dictionary)
    # saver.save(sess, f"{log_path}/model.ckpt")

# snapshotter = Snapshotter()
# with TFTrainer(snapshotter) as trainer:
#     trainer.restore('data/local/experiment/')
#     trainer.resume(n_epochs=500, batch_size=4000)
#     env = gym.make('InvertedDoublePendulum-v2')
#     for repeat in range(3):
#         ob = env.reset()
#         policy = trainer._algo.policy
#         policy.reset()
#         imgs = []
#         for timestep in range(1000):
#             ob, rew, done, info = env.step(policy.get_action(ob)[0])
#             imgs.append(env.render('rgb_array'))
#         save_video(imgs, os.path.join(f"policy_videos/skill_{repeat}.avi"))
#     env.close()
