import os

import pickle
from datetime import datetime
import itertools

import statistics
import numpy as np
import tensorflow as tf
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.dmsrd_punish import ReLUModel, AIRLMultiStyleDynamic, new_likelihood, safe_log_np, grid_objective
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_tree(x, xp, k):
    # https://github.com/BiuBiuBiLL/NPEET_LNC/blob/master/lnc.py
    # https://github.com/scipy/scipy/issues/9890 p=2 or np.inf
    tree = cKDTree(x)
    return tree.query(xp, k=k + 1, p=float('inf'))[0][:, k] # chebyshev distance of k+1-th nearest neighbor


def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=100, repeat_each_skill=3,
                                                     separate_styles=True)

    # skills = [0, 3, 8, 5, 7, 2, 6, 9, 1, 4]
    skills = [3, 7, 5, 6, 0, 8, 1, 9, 4, 2]
    mixtures = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.2867442413870225, 0.24589425481643865, 0.3790942546075972, 0.08558681480083458, 0.0026804343881068546, 0.0, 0.0], [0.23640026758018873, 0.31894649749504245, 0.06299201269156939, 0.12240180673606574, 0.25925941549713377, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.2553869895739193, 0.10755857151347319, 0.17168187162781626, 0.22895981340464752, 0.1075737771037049, 0.12883897677643896, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

    experts_multi_styles = [demonstrations[i] for i in skills]
    num_skills = 7

    # Hyperparameters
    state_only = True
    deterministic = False
    n_timesteps = 1000
    # load_path = f"data/inverted_pendulum_dmsrd_nomip/20_09_2021_22_10_40"
    load_path = f"data/inverted_pendulum_dmsrd_distance/16_10_2021_21_13_47"
    assert os.path.exists(load_path), "load path does not exist! "

    task_reward = ReLUModel("task", env.observation_space.shape[0]) \
        if state_only \
        else ReLUModel("task", env.observation_space.shape[0] + env.action_space.shape[0])

    strategy_rewards = []

    skill_vars = []
    value_vars = []
    policies = []
    value_fs = []
    iteration = 0
    save_dictionary = {}
    for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='task')):
        save_dictionary[f'my_task_{idx}'] = var

    task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task')
    while iteration < num_skills:
        new_skill_reward = ReLUModel(f"skill_{len(strategy_rewards)}", env.observation_space.shape[0])

        policy = GaussianMLPPolicy(name=f'policy_{len(strategy_rewards)}', env_spec=env.spec,
                                   hidden_sizes=(32, 32))

        value_fn = ReLUModel(f"value_{len(strategy_rewards)}", env.spec.observation_space.flat_dim)

        skill_vars.append(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'skill_{len(strategy_rewards)}'))
        # value_vars.append(
        #     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'value_{len(strategy_rewards)}'))

        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{len(strategy_rewards)}')):
            save_dictionary[f'my_skill_{len(strategy_rewards)}_{idx}'] = var
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{len(strategy_rewards)}')):
            save_dictionary[f'my_policy_{len(strategy_rewards)}_{idx}'] = var
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'value_{len(strategy_rewards)}')):
            save_dictionary[f'my_value_{len(strategy_rewards)}_{idx}'] = var

        policies.append(policy)
        strategy_rewards.append(new_skill_reward)
        value_fs.append(value_fn)
        iteration+=1

    num_skills = len(strategy_rewards)
    reward_fs = []
    algos = []
    for skill in range(num_skills):
        with tf.variable_scope(f"iter_{iteration}_skill_{skill}"):
            irl_var_list = task_vars + skill_vars[skill]
            irl_model = AIRLMultiStyleDynamic(env, task_reward,
                                              strategy_rewards[skill],
                                              value_fs[skill],
                                              expert_trajs=list(itertools.chain(*demonstrations)),
                                              var_list=irl_var_list,
                                              state_only=state_only,
                                              fusion=True,
                                              l2_reg_skill=0.001,
                                              l2_reg_task=0.0001,
                                              max_itrs=10)

            reward_weights = [0.0] * num_skills
            reward_weights[skill] = 1.0
            algo = IRLTRPO(
                reward_weights=reward_weights.copy(),
                env=env,
                policy=policies[skill],
                irl_model=irl_model,
                n_itr=1500,  # doesn't matter, will change
                batch_size=32,
                max_path_length=1000,
                discount=0.99,
                store_paths=False,
                discrim_train_itrs=10,
                irl_model_wt=1.0,
                entropy_weight=0.1,  # from AIRL: this should be 1.0 but 0.1 seems to work better
                zero_environment_reward=True,
                baseline=LinearFeatureBaseline
            )
            reward_fs.append(irl_model)
            algos.append(algo)

    trajectories = []
    for i in range(20):
        with open(f'data/train_trajs/trajectories_{i}.pkl', "rb") as f:
            trajectories.extend(pickle.load(f))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{load_path}/model.ckpt")

        with rllab_logdir(algo=algo, dirname=load_path + '/reward'):
        #     for demo in trajectories:
        #         traj=demo["observations"]
        #         nobs = traj[1:]
        #         nobs = np.r_[nobs, 0.0*np.expand_dims(np.ones_like(nobs[0]), axis=0)]
        #         demo["actions"] = np.array(demo["actions"]).squeeze(axis=2)
        #         for pol in range(len(policies)):
        #             probs = irl_model.eval_expert_probs([demo], policies[pol], insert=False)
        #             reward = tf.get_default_session().run(reward_fs[pol].discrim_output,
        #                                                 feed_dict={reward_fs[pol].obs_t: traj, reward_fs[pol].nobs_t: nobs,
        #                                                 reward_fs[pol].lprobs: np.expand_dims(probs[0], axis=1)})
        #             score = reward[:, 0]
        #             logger.record_tabular(f'StrategyReward_{pol}', np.mean(score))
        #         logger.dump_tabular(with_prefix=False, write_header=True)
        #     strat = 0
        #     score_sum = 0
        #     for idx, traj in enumerate(trajectories):
        #         reward_cent = tf.get_default_session().run(reward_fs[0].reward_task, feed_dict={reward_fs[0].obs_t: traj})
        #         score = reward_cent[:, 0]
        #         logger.record_tabular(f'Center', np.sum(score))
        #
        #         reward = tf.get_default_session().run(reward_fs[0].reward_skill_weighted,
        #                                               feed_dict={reward_fs[0].obs_t: traj, reward_fs[0].weights: np.repeat([mixtures[strat]], len(traj), axis=0)})
        #         score = reward[:, 0]
        #         score_sum += np.sum(score)
        #         if idx>0 and (idx+1) % 10 == 0:
        #             logger.record_tabular(f'DemoReward_{skills[strat]}', np.mean(score_sum))
        #             score_sum = 0
        #         if idx>0 and (idx+1) % 200 == 0:
        #             strat += 1
        #         logger.dump_tabular(with_prefix=False, write_header=False)
            for traj in trajectories:
                reward_cent = tf.get_default_session().run(reward_fs[0].reward_task, feed_dict={reward_fs[0].obs_t: traj})
                score = reward_cent[:, 0]
                logger.record_tabular(f'Center', np.mean(score))

                for strat in range(num_skills):
                    reward = tf.get_default_session().run(reward_fs[strat].reward_skill,
                                                          feed_dict={reward_fs[strat].obs_t: traj})
                    score = reward[:, 0]
                    logger.record_tabular(f'DemoReward_{skills[strat]}', np.mean(score))
                logger.dump_tabular(with_prefix=False, write_header=False)
        # with rllab_logdir(algo=algo, dirname=load_path + '/demo_reward'):
        #     rew = []
        #     for demo in experts_multi_styles:
        #         strat_rew = []
        #         for strat in range(len(reward_fs)):
        #             rew_repeat = 0
        #             for traj in demo:
        #                 reward = tf.get_default_session().run(reward_fs[strat].reward_skill,
        #                                                       feed_dict={reward_fs[strat].obs_t: traj["observations"]})
        #                 score = np.mean(reward[:, 0])
        #                 rew_repeat += np.mean(score)
        #             strat_rew.append(rew_repeat)
        #             # logger.record_tabular(f'Strategy{strat}', strat_rew)
        #         rew.append(strat_rew)
        #     logger.dump_tabular(with_prefix=False, write_header=True)
        #
        # rew = np.array(rew)
        # for j in range(len(rew[0])):
        # # rew[:, j] = rew[:, j]/np.linalg.norm(rew[:, j])
        #     rew[:, j] = (rew[:, j] - np.min(rew[:, j]))/np.ptp(rew[:, j])
        #
        # # for i in range(len(rew)):
        # #     #   rew[i] = rew[i]/np.linalg.norm(rew[i])
        # #     rew[i] = (rew[i] - np.min(rew[i])) / np.ptp(rew[i])
        #
        # name = [f'Demonstration {i}' for i in range(len(rew))]
        # trajectories = [f'Strategy {i}' for i in range(len(rew[0]))]
        #
        # fig, ax = plt.subplots()
        #
        # im, cbar = heatmap(rew, name, trajectories, ax=ax,
        #                    cmap="YlGn", cbarlabel="reward")
        # texts = annotate_heatmap(im)
        #
        # fig.tight_layout()
        # plt.savefig(f'{load_path}/heatmap.png')
        # plt.close()
        #
        # rew_nomix = rew[[0,1,2,3,4,7,9]]
        #
        # name = [f'Demonstration {i}' for i in [0,1,2,3,4,7,9]]
        # trajectories = [f'Strategy {i}' for i in range(len(rew[0]))]
        #
        # fig, ax = plt.subplots()
        #
        # im, cbar = heatmap(rew_nomix, name, trajectories, ax=ax,
        #                    cmap="YlGn", cbarlabel="reward")
        # texts = annotate_heatmap(im)
        #
        # fig.tight_layout()
        # plt.savefig(f'{load_path}/heatmap_nomix.png')
        # plt.close()
        #
        # with rllab_logdir(algo=algo, dirname=load_path + '/probs'):
        #     for demonst in range(len(experts_multi_styles)):
        #         post_likelihoods = np.array([reward_fs[0].eval_expert_probs(experts_multi_styles[demonst], policies[i])
        #                       for i in range(len(policies))])
        #         logger.record_tabular(f'Demonstration Log Likelihood {demonst}', grid_objective([mixtures[demonst]],
        #                                                                 np.resize(np.exp(post_likelihoods),
        #                                                                           (
        #                                                                               post_likelihoods.shape[
        #                                                                                   0],
        #                                                                               post_likelihoods.shape[
        #                                                                                   1] *
        #                                                                               post_likelihoods.shape[
        #                                                                                   2]))))
        #     logger.dump_tabular(with_prefix=False, write_header=True)

        # with rllab_logdir(algo=algo, dirname=load_path + '/probs'):
        #     for demo_ind in skills:
        #         post_rew = get_reward(env, mixtures[demo_ind], policies)
        #         divergence = get_divergence(env, mixtures[demo_ind], policies, experts_multi_styles[demo_ind])
        #         logger.record_tabular(f"Demonstration {demo_ind}", post_rew)
        #         logger.record_tabular(f"Divergence {demo_ind}", divergence)
        #     logger.dump_tabular(with_prefix=False, write_header=True)
        #     for idx, policy in enumerate(policies):
        #         post_likelihoods = new_likelihood(np.array(reward_fs[0].eval_expert_probs(demonstrations[skills[idx]], policy)))
        #         post_probs = new_likelihood(np.array(reward_fs[0].eval_numerical_integral(demonstrations[skills[idx]], policy)))
        #         logger.record_tabular(f"Policy Likelihood{idx}", post_likelihoods)
        #         logger.record_tabular(f"Policy LogProb{idx}", post_probs)
        #     logger.dump_tabular(with_prefix=False, write_header=True)

        # with rllab_logdir(algo=algo, dirname=load_path + '/probs'):
        #     for strat in range(len(skills)):
        #         post_probs = new_likelihood(np.array(reward_fs[0].eval_expert_mix(demonstrations[strat], policies, mixtures[strat])))
        #         logger.record_tabular(f"Mixture LogProb {skills[strat]}", post_probs)
        #     logger.dump_tabular(with_prefix=False, write_header=True)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_reward(env, mixture, policies):
    episode_return = 0.0
    for _ in range(10):
        ob = env.reset()
        for timestep in range(1000):
            act = np.dot(mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            episode_return += rew
    return episode_return/10


def get_divergence(env, mixture, policies, demonstration):
    dist = 0.0
    for _ in range(10):
        obs = []
        ob = env.reset()
        for timestep in range(1000):
            act = np.dot(mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            obs.append(ob)

        expert = demonstration[0]["observations"]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 3)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    return dist/10


if __name__ == "__main__":
    main()
