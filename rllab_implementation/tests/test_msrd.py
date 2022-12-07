import os
import csv
import pickle
from datetime import datetime
import tensorflow as tf
import numpy as np
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.msd import ReLUModel, AIRLMultiStyleSingle
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
from inverse_rl.utils.likelihood_utils import new_likelihood
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

    # load expert demonstration data from this repo's data generation
    # experts = load_latest_experts('data/inverted_pendulum', n=5, min_return=-10)

    # load expert demonstration data from core_MSD's data generation
    # experts_multi_styles = load_expert_from_core_MSD('data/InvertedPendulum10skills.pkl', length=1000,  repeat_each_skill=3,
    #                                                  separate_styles=True)


    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000,  repeat_each_skill=3,
                                                     separate_styles=True)
    # order = [3, 7, 5, 4, 0, 8, 1, 9, 6, 2]
    experts_multi_styles = demonstrations

    # Hyperparameters
    state_only = True
    num_skills = len(experts_multi_styles)
    discriminator_update_step = 10
    now = datetime.now()
    log_path = f"data/inverted_pendulum_msrd/16_03_2022_19_35_01"

    save_dictionary = {}

    center_reward = ReLUModel("center", env.observation_space.shape[0]) \
        if state_only \
        else ReLUModel("center", env.observation_space.shape[0] + env.action_space.shape[0])

    for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='center')):
        save_dictionary[f'my_center_{idx}'] = var

    n_timesteps = 1000
    deterministic = False

    reward_fs = []
    policies = []
    algos = []
    for skill in range(num_skills):
        irl_model = AIRLMultiStyleSingle(env, center_reward,
                                         expert_trajs=experts_multi_styles[skill],
                                         state_only=state_only, fusion=True, max_itrs=discriminator_update_step, name=f'skill_{skill}')

        for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'skill_{skill}')):
            save_dictionary[f'my_skill_{skill}_{idx}'] = var
        policy = GaussianMLPPolicy(name=f'policy_{skill}', env_spec=env.spec, hidden_sizes=(32, 32))
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'policy_{skill}')):
            save_dictionary[f'my_policy_{skill}_{idx}'] = var
        algo = IRLTRPO(
            env=env,
            policy=policy,
            irl_model=irl_model,
            n_itr=1500,  # doesn't matter, will change
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
        policies.append(policy)
        reward_fs.append(irl_model)
        algos.append(algo)

    ground_truths = []
    with open('data/GroundTruthReward.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first = True
        for row in reader:
            if first:
                first = False
                continue
            ground_truths.append(float(row[0]))

    trajectories = []
    for i in range(20):
        with open(f'data/trajs/trajectories_{i}.pkl', "rb") as f:
            trajectories.extend(pickle.load(f))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model.ckpt")

        with rllab_logdir(algo=algo, dirname=log_path + '/reward'):
            task_rewards = []
            for traj in trajectories:
                reward_cent = tf.get_default_session().run(reward_fs[0].reward_center, feed_dict={reward_fs[0].obs_t: traj})
                score = reward_cent[:, 0]
                task_rewards.append(np.mean(score))
                logger.record_tabular(f'Center', np.mean(score))

                for pol in range(len(policies)):
                    reward = tf.get_default_session().run(reward_fs[pol].reward_peri,
                                                          feed_dict={reward_fs[pol].obs_t: traj})
                    score = reward[:, 0]
                    logger.record_tabular(f'StrategyReward_{pol}', np.mean(score))
                logger.dump_tabular(with_prefix=False, write_header=False)
            logger.record_tabular(f'Correlation', np.corrcoef(ground_truths, task_rewards))
            logger.dump_tabular(with_prefix=False, write_header=True)

        rew = []
        for demo in experts_multi_styles:
            strat_rew = []
            for strat in range(len(reward_fs)):
                rew_repeat = 0
                for traj in demo:
                    reward = tf.get_default_session().run(reward_fs[strat].reward_peri,
                                                          feed_dict={reward_fs[strat].obs_t: traj["observations"]})
                    score = np.mean(reward[:, 0])
                    rew_repeat += np.mean(score)
                strat_rew.append(rew_repeat)
            rew.append(strat_rew)

        rew = np.array(rew)
        for j in range(len(rew[0])):
        # rew[:, j] = rew[:, j]/np.linalg.norm(rew[:, j])
            rew[:, j] = (rew[:, j] - np.min(rew[:, j]))/np.ptp(rew[:, j])

        # for i in range(len(rew)):
        #     #   rew[i] = rew[i]/np.linalg.norm(rew[i])
        #     rew[i] = (rew[i] - np.min(rew[i])) / np.ptp(rew[i])

        name = [f'Demonstration {i}' for i in range(len(rew))]
        trajectories = [f'Strategy {i}' for i in range(len(rew[0]))]

        fig, ax = plt.subplots()

        im, cbar = heatmap(rew, name, trajectories, ax=ax,
                           cmap="Blues", cbarlabel="reward")
        # texts = annotate_heatmap(im)

        fig.tight_layout()
        plt.savefig(f'{log_path}/heatmap.png')
        plt.close()

        post_likelihoods = [new_likelihood(np.array(reward_fs[0].eval_expert_probs(demonstrations[i], policies[i], fix=True)))
                      for i in range(len(experts_multi_styles))]

        with rllab_logdir(algo=algo, dirname=log_path + '/probs'):
            rewards = []
            divergences = []
            for demo_ind in range(len(policies)):
                post_rew = get_reward(env, policies[demo_ind])
                divergence = get_divergence(env, policies[demo_ind], experts_multi_styles[demo_ind], log_path, demo_ind)
                rewards.append(post_rew)
                divergences.append(divergence)
            logger.record_tabular(f"Rewards", rewards)
            logger.record_tabular(f"Divergences", divergences)
            logger.record_tabular(f'Final_Likelihoods', post_likelihoods)
            logger.record_tabular(f"Mean_Rewards", np.mean(rewards))
            logger.record_tabular(f"Mean_Divergences", np.mean(divergences))
            logger.record_tabular(f'Mean_Likelihoods', np.mean(post_likelihoods))
            logger.dump_tabular(with_prefix=False, write_header=True)


def get_reward(env, policy):
    episode_return = 0.0
    for _ in range(10):
        ob = env.reset()
        for timestep in range(1000):
            act = policy.get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            episode_return += rew
    return episode_return/10


def get_divergence(env, policy, demonstration, log_path, skill_index):
    dist = 0.0
    trajs = []
    for _ in range(10):
        obs = []
        ob = env.reset()
        for timestep in range(1000):
            act = policy.get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            obs.append(ob)

        trajs.append(obs)
        expert = demonstration[0]["observations"]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 3)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    with open(f"{log_path}/likelihood/trajs_skill_{skill_index}.pkl", "wb") as f:
        pickle.dump(trajs, f)
    return dist/10


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

    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.labelweight'] = 'bold'
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
    ax.set_xticklabels(col_labels, fontweight='bold', fontsize=13.0)
    ax.set_yticklabels(row_labels, fontweight='bold', fontsize=13.0)

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


if __name__ == "__main__":
    main()
