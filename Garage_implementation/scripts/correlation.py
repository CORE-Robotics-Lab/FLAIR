import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


def normalize_and_rescale(to_normalize_arr, target_arr):
    target_max = np.max(target_arr)
    target_min = np.min(target_arr)
    to_normalize_max = np.max(to_normalize_arr)
    to_normalize_min = np.min(to_normalize_arr)
    normalized = (to_normalize_arr - to_normalize_min) / (to_normalize_max - to_normalize_min)
    rescaled = normalized * (target_max - target_min) + target_min
    return rescaled


if __name__ == "__main__":
    # dataframe = pd.read_csv('data/TestRewardComparison.csv')
    dataframe = pd.read_csv('data/BipedalTestRewardComparison.csv')

    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1
    fig, ax = plt.subplots(1, 1, figsize=(6, 5)) #(13,4)

    dataframe["Ground Truth Reward"] = dataframe["Ground Truth"]
    dataframe["Center"] = dataframe["MSRD Center"]

    dataframe["DMSRD Rescaled Task Reward"] = normalize_and_rescale(dataframe["DMSRD Task Reward"], dataframe["Ground Truth Reward"])
    dataframe["Center"] = normalize_and_rescale(dataframe["Center"], dataframe["Ground Truth Reward"])
    dataframe["AIRL Batch"] = normalize_and_rescale(dataframe["AIRL Batch"], dataframe["Ground Truth Reward"])

    # sns.lmplot(x="Ground Truth Reward", y="DMSRD Rescaled Task Reward", data=dataframe)
    slope, intercept, r_value, p_value, std_err = linregress(dataframe["Ground Truth Reward"], dataframe["AIRL Batch"])
    ax.scatter(dataframe["Ground Truth Reward"], dataframe["AIRL Batch"],s=1.5,label=r'AIRL Batch: $r=0.281$', color='green')

    # ax.plot(dataframe["Ground Truth Reward"], intercept + slope * dataframe["Ground Truth Reward"], 'r', color="black")
    # ax.set_title('AIRL', fontsize=14)
    # ax.legend()
    # legend1 = ax.legend(*scatter.legend_elements(num=3),
    #                     loc="upper left")
    # ax.add_artist(legend1)
    # ax.legend([f'R = {round(r_value,3)}'])
    ax.set_xlabel('Ground-Truth Task Reward')
    ax.set_ylabel('Rescaled Estimated Reward')
    slope, intercept, r_value, p_value, std_err = linregress(dataframe["Ground Truth Reward"], dataframe["Center"])
    ax.scatter(dataframe["Ground Truth Reward"], dataframe["Center"],s=1.5,label=r"MSRD: $r=0.401$", color='yellow')

    # ax.plot(dataframe["Ground Truth Reward"], intercept + slope * dataframe["Ground Truth Reward"], 'r', color="black")
    # ax.legend([f'R = {round(r_value,3)}'])
    # ax.set_title('MSRD', fontsize=14)
    ax.set_xlabel('Ground-Truth Task Reward')
    ax.set_ylabel('Rescaled Estimated Reward')
    slope, intercept, r_value, p_value, std_err = linregress(dataframe["Ground Truth Reward"], dataframe["DMSRD Rescaled Task Reward"])
    ax.scatter(dataframe["Ground Truth Reward"], dataframe["DMSRD Rescaled Task Reward"],s=1.5,label=r"DMSRD: $r=0.582$", color='tab:blue')

    # ax.plot(dataframe["Ground Truth Reward"], intercept + slope * dataframe["Ground Truth Reward"], 'r', color="black")
    # ax.legend([f'R = {round(r_value,3)}'])

    # ax.plot(range(-45000, 37000, 1), range(-45000, 37000, 1), color="black", linestyle="--")
    ax.plot(range(-1200, 330, 1), range(-1200, 330, 1), color="black", linestyle="--")
    lgnd = ax.legend(prop={'weight':'bold'})
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]
    lgnd.legendHandles[2]._sizes = [30]
    # ax.set_title('DMSRD', fontsize=14)
    ax.set_xlabel('Ground-Truth Task Reward')
    ax.set_ylabel('Rescaled Estimated Reward')

    fig.tight_layout(pad=2.0)
    # plt.plot(range(-1500, 0, 1), range(-1500, 0, 1), color="black", linestyle="--")
    # ax.xaxis.set_tick_params(labelsize=14)
    # ax.yaxis.set_tick_params(labelsize=15)
    # fig.text(0.5, 0.04, 'Ground Truth Reward', ha='center', fontsize=18)
    # fig.text(0.04, 0.5, 'Rescaled Estimated Reward', va='center', rotation='vertical', fontsize=18)
    plt.savefig("data/figs/task_correlation_walker.png")
    plt.close(fig)
