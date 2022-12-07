#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.
Here it runs CartPole-v1 environment with 100 iterations.
Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
from datetime import datetime
import gym
import dowel
from dowel import logger
import tensorflow as tf
from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from airl.irl_trpo import TRPO
from models.msd import ReLUModel, AIRLMultiStyleSingle

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
from airl.test_performance import *
import matplotlib
import matplotlib.pyplot as plt
import csv

now = datetime.now()
log_path = f"data/bipedal_walker_msrd_efficiency/{now.strftime('%d_%m_%Y_%H_%M_%S')}"

irl_models = []
policies = []
algos = []
trainers = []

demonstrations = load_expert_from_core_MSD(
    'data/BipedalWalker10skills.pkl', length=1000,
    repeat_each_skill=3,
    separate_styles=True)
env = GymEnv('BipedalWalker-v3')

n_epochs = 2000
timesteps = 1000

dmsrd_truths = [-109242.3585,-41093.92305,-41264.78632,-34722.68592,-35931.18207,-50204.55801,-39185.67776,-55735.6289,-56906.34516,-55006.19801]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}

    center_reward = ReLUModel("center", env.observation_space.shape[0])

    for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='center')):
        save_dictionary[f'my_center_{idx}'] = var
    for index in range(len(demonstrations)):
        snapshotter = Snapshotter(f'{log_path}/skill_{index}')
        trainer = Trainer(snapshotter)

        irl_model = AIRLMultiStyleSingle(env, center_reward,
                                         expert_trajs=demonstrations[index],
                                         state_only=True, fusion=True, max_itrs=10, name=f'skill_{index}')
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
                               max_episode_length=timesteps,
                               is_tf_worker=True,
                             n_workers=10)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    index=index,
                    center_grads=True,
                    sampler=sampler,
                    irl_model=irl_model,
                    generator_train_itrs=1,
                    discrim_train_itrs=10,
                    discount=0.99,
                    max_kl_step=0.01)
        trainer.setup(algo, env)
        trainer._start_worker()
        trainers.append(trainer)
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    sess.run(tf.global_variables_initializer())

    with open(f'{log_path}/efficiency.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Likelihoods"])
    with open(f'{log_path}/efficiency_epochs.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Epochs"])

    likelihoods_epoch = []
    threshold_epoch = [[0] for _ in range(len(demonstrations))]
    first_epoch = [True for _ in range(len(demonstrations))]
    for epoch in range(n_epochs):
        center_reward_gradients = None
        for i in range(len(demonstrations)):
            trainer = trainers[i]
            logger.remove_all()
            tabular_log_file = os.path.join(trainer._snapshotter.snapshot_dir, 'progress.csv')
            logger.add_output(dowel.CsvOutput(tabular_log_file))
            logger.add_output(dowel.StdOutput())
            trainer.train(n_epochs=epoch+1, batch_size=10000, start_epoch=epoch)
            if center_reward_gradients is None:
                center_reward_gradients = algos[
                    i].center_reward_gradients
            else:
                assert center_reward_gradients.keys() == algos[
                    i].center_reward_gradients.keys()
                for key in center_reward_gradients.keys():
                    center_reward_gradients[key] += \
                    algos[i].center_reward_gradients[key]
        feed_dict = {}
        assert center_reward.grad_map_vars.keys() == center_reward_gradients.keys()
        for key in center_reward.grad_map_vars.keys():
            feed_dict[center_reward.grad_map_vars[key]] = \
            center_reward_gradients[key]
        sess.run(center_reward.step, feed_dict=feed_dict)

        likelihood = get_likelihoods(irl_model, demonstrations, policies)
        likelihoods_epoch.append(likelihood)
        for i in range(len(likelihood)):
            if epoch > 30 and first_epoch[i] and likelihood[i] > dmsrd_truths[i]:
                threshold_epoch[i].append(epoch)
                first_epoch = False

    with open(f'{log_path}/efficiency.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(likelihoods_epoch)
    with open(f'{log_path}/efficiency_epochs.csv', 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(threshold_epoch)

    trainers[0]._shutdown_worker()

    env_test = gym.make('BipedalWalker-v3')
    # for i in range(len(demonstrations)):
    #     ob = env_test.reset()
    #     policy = policies[i]
    #     imgs = []
    #     for timestep in range(timesteps):
    #         ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
    #         imgs.append(env_test.render('rgb_array'))
    #     save_video(imgs, os.path.join(f"{log_path}/policy_videos/skill_{i}.avi"))
    # env_test.close()

    with open(f'{log_path}/likelihood.csv', 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        likelihoods = get_likelihoods(irl_model, demonstrations, policies)
        rewards = get_reward(env_test, policies, timesteps)
        divergences = get_divergence(env_test, policies, demonstrations, timesteps)
        csvwriter.writerow(["Likelihoods"])
        csvwriter.writerow(likelihoods)
        csvwriter.writerow(["Rewards"])
        csvwriter.writerow(rewards)
        csvwriter.writerow(["Divergences"])
        csvwriter.writerow(divergences)
        csvwriter.writerow(["Mean_Likelihoods"])
        csvwriter.writerow([np.mean(likelihoods)])
        csvwriter.writerow(["Mean_Rewards"])
        csvwriter.writerow([np.mean(rewards)])
        csvwriter.writerow(["Mean_Divergences"])
        csvwriter.writerow([np.mean(divergences)])

    trajectories = []
    for i in range(40):
        with open(f'data/biwalker_trajs/trajectories_{i}.pkl', "rb") as f:
            trajectories.extend(pickle.load(f))

    ground_truths = []
    with open(f'data/BipedalTestReward.csv',
              newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first = True
        for row in reader:
            if first:
                first = False
                continue
            ground_truths.append(float(row[0]))

    record = np.zeros(len(trajectories))
    for tidx, traj in enumerate(trajectories):
        reward_cent = tf.get_default_session().run(
            irl_model.reward_center,
            feed_dict={irl_model.obs_t: traj["observations"]})
        score = reward_cent[:, 0]
        record[tidx] = np.mean(score)

    with open(f'{log_path}/reward.csv', 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Reward correlation"])
        csvwriter.writerow(np.corrcoef(ground_truths, record).tolist())
        csvwriter.writerow(["Task Reward"])
        csvwriter.writerow(record.tolist())

    rew = []
    for demo in demonstrations:
        strat_rew = []
        for strat in range(len(irl_models)):
            rew_repeat = 0
            for traj in demo:
                reward = tf.get_default_session().run(
                    irl_models[strat].reward_peri,
                    feed_dict={
                        irl_models[strat].obs_t: traj["observations"]})
                score = np.mean(reward[:, 0])
                rew_repeat += np.mean(score)
            strat_rew.append(rew_repeat)
        rew.append(strat_rew)

    rew = np.array(rew)
    for j in range(len(rew[0])):
        rew[:, j] = (rew[:, j] - np.min(rew[:, j])) / np.ptp(rew[:, j])

    name = [f'Demonstration {i}' for i in range(len(rew))]
    trajectories = [f'Strategy {i}' for i in range(len(rew[0]))]

    fig, ax = plt.subplots()

    im, cbar = heatmap(rew, name, trajectories, ax=ax,
                       cmap="YlGn", cbarlabel="reward")
    texts = annotate_heatmap(im)

    fig.tight_layout()
    plt.savefig(f'{log_path}/heatmap.png')
    plt.close()

    saver = tf.train.Saver(save_dictionary)
    saver.save(sess, f"{log_path}/model.ckpt")
