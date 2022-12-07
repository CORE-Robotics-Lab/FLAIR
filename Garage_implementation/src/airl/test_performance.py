import numpy as np
from scipy.spatial import cKDTree
from models.dmsrd import new_likelihood, grid_objective
import pickle
import tensorflow as tf
import csv
import matplotlib
import matplotlib.pyplot as plt


def safe_log_np(x):
    x = np.clip(x, 1e-37, None, dtype=np.float64)
    return np.log(x, dtype=np.float64)

def grid_objective(weights, action_prob):
    out = np.matmul(weights, action_prob, dtype=np.float64)
    out = np.sum(safe_log_np(out), axis=1, dtype=np.float64)
    return out

# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_tree(x, xp, k):
    # https://github.com/BiuBiuBiLL/NPEET_LNC/blob/master/lnc.py
    # https://github.com/scipy/scipy/issues/9890 p=2 or np.inf
    tree = cKDTree(x)
    return np.clip(tree.query(xp, k=k + 1, p=float('inf'))[0][:, k], 1e-30, None) # chebyshev distance of k+1-th nearest neighbor


def kldiv(x, xp, k=3):
    """ KL Divergence between p and q for x~p(x), xp~q(x)
    """
    x, xp = np.asarray(x), np.asarray(xp)
    assert k < min(x.shape[0], xp.shape[0]), "Set k smaller than num. samples - 1"
    assert x.shape[1] == xp.shape[1], "Two distributions must have same dim."

    x, xp = x.reshape(x.shape[0], -1), xp.reshape(xp.shape[0], -1)
    n, d = x.shape
    m, _ = xp.shape
    x = add_noise(x)  # fix np.log(0)=inf issue

    const = np.log(m) - np.log(n - 1)
    nn = query_tree(x, x, k)
    nnp = query_tree(xp, x, k - 1)  # (m, k-1)
    return const + d * (np.log(nnp).mean() - np.log(nn).mean())


def get_reward(env, policies, timesteps):
    returns = []
    for policy in policies:
        episode_return = 0.0
        for _ in range(10):
            ob = env.reset()
            for timestep in range(timesteps):
                act = policy.get_action(ob)[0]
                act_executed = act
                ob, rew, done, info = env.step(act_executed)
                episode_return += rew
        returns.append(episode_return/10)
    return returns


def get_divergence(env, policies, demonstrations, timesteps):
    dists = []
    for demo_ind in range(len(demonstrations)):
        dist = 0.0
        # trajs = []
        expert = np.concatenate([demo["observations"] for demo in demonstrations[demo_ind]])
        for _ in range(10):
            obs = []
            ob = env.reset()
            for timestep in range(timesteps):
                act = policies[demo_ind].get_action(ob)[0]
                act_executed = act
                ob, rew, done, info = env.step(act_executed)
                obs.append(ob)

            # trajs.append(obs)
            dist += kldiv(expert, obs)
    return dists


def get_batch_rewdiv(env, policy, demonstrations, timesteps):
    dists = []
    rews = []
    for i in range(len(demonstrations)):
        episode_return = 0.0
        obs = []
        ob = env.reset()
        for timestep in range(timesteps):
            act = policy.get_action(ob)[0]
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            obs.append(ob)
            episode_return += rew
        expert = np.concatenate([demo["observations"] for demo in demonstrations[i]])
        rews.append(episode_return)
        dist = kldiv(expert, obs)
        dists.append(dist)
    return rews, dists


def get_batch_likelihoods(reward, demonstrations, policy):
    return [new_likelihood(np.array(reward.eval_expert_probs(demonstrations[i], policy, fix=True)))
                      for i in range(len(demonstrations))]


def get_dmsrd_likelihood(demonstrations, policies, irl_model, mixtures):
    likels = []
    for demonst in range(len(demonstrations)):
        post_likelihoods = new_likelihood(irl_model.eval_mixture_probs(demonstrations[demonst], mixtures[demonst], policies, fix=True))
        likels.append(post_likelihoods)

    # for demonst in range(len(demonstrations)):
    #     post_likelihoods = np.array([irl_model.eval_expert_probs(demonstrations[demonst], policies[i], fix=True)
    #                   for i in range(len(policies))])
    #     likels.append(grid_objective([mixtures[demonst]],
    #                                                             np.resize(np.exp(post_likelihoods),
    #                                                                       (
    #                                                                           post_likelihoods.shape[
    #                                                                               0],
    #                                                                           post_likelihoods.shape[
    #                                                                               1] *
    #                                                                           post_likelihoods.shape[
    #                                                                               2]))))
    return likels


def get_dmsrd_divergence(env, mixtures, policies, demonstrations, timesteps):
    dists = []
    rews = []
    for demo_ind in range(len(demonstrations)):
        dist = 0.0
        episode_return = 0.0
        mixture = mixtures[demo_ind]
        # trajs = []
        expert = np.concatenate([demo["observations"] for demo in demonstrations[demo_ind]])
        for _ in range(10):
            obs = []
            ob = env.reset()
            for timestep in range(timesteps):
                act = np.dot(mixture, [policy.get_action(ob)[0] for policy in policies])
                act_executed = act
                ob, rew, done, info = env.step(act_executed)
                obs.append(ob)
                episode_return += rew

            # trajs.append(obs)
            dist += kldiv(expert, obs)

        # with open(f"trajs/dmsrd/0/trajs_skill_{demo_ind}.pkl", "wb") as f:
        # with open(f"trajs/dmsrd/2/trajs_skill_{demo_ind}.pkl", "wb") as f:
        #     pickle.dump(trajs, f)
        rews.append(episode_return/10)
        dists.append(dist/10)
    return rews, dists


def get_likelihoods(reward, demonstrations, policies):
    return [new_likelihood(np.array(reward.eval_expert_probs(demonstrations[i], policies[i], fix=True)))
                      for i in range(len(demonstrations))]


def plot_reward(irl_model, log_path):
    trajectories = []
    for i in range(20):
        with open(f'data/trajs/trajectories_{i}.pkl', "rb") as f:
            trajectories.extend(pickle.load(f))

    record = np.zeros(len(trajectories))
    for tidx, traj in enumerate(trajectories):
        reward = tf.get_default_session().run(irl_model.reward,
                                              feed_dict={irl_model.obs_t: traj})
        score = reward[:, 0]
        record[tidx] = np.mean(score)

    with open(f'{log_path}/reward.csv', 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Task Reward"])
        csvwriter.writerows(record.to_list())


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
    im = ax.imshow(data, vmin=0, vmax=1, **kwargs)

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
