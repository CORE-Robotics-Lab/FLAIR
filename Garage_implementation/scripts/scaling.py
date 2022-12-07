import numpy as np
import matplotlib.pyplot as plt

def normalize_and_rescale(to_normalize_arr, target_arr):
    target_max = np.max(target_arr)
    target_min = np.min(target_arr)
    to_normalize_max = np.max(to_normalize_arr)
    to_normalize_min = np.min(to_normalize_arr)
    normalized = (to_normalize_arr - to_normalize_min) / (to_normalize_max - to_normalize_min)
    rescaled = normalized * (target_max - target_min) + target_min
    return rescaled


if __name__ == "__main__":
    data = np.genfromtxt('data/lunar_lander_dmsrd_scale/2022_05_26_10_35_22/scale.csv', delimiter=',')
    # data = np.genfromtxt('data/biwalker_dmsrd_scale/2022_05_14_11_13_28/scale.csv', delimiter=',')

    plt.rcParams['axes.labelsize'] = 15 # 17
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(1, 4, figsize=(19, 4)) #(6,4)

    demonstration = np.arange(0, 100, 100/len(data))

    ax[0].axhline(y=-6346.6, color='black', linestyle='--', lw=3, label='FLAIR on 10 Demos')
    ax[0].axhline(y=-7418.1, color='green', linestyle='--', lw=3, label='AIRL on 10 Demos')
    ax[0].axhline(y=-9895.3, color='yellow', linestyle='--', lw=3, label='MSRD on 10 Demos')
    ax[0].plot(demonstration, data[:,0],'-o') #,label=r'AIRL Batch: $r=0.776$')
    # ax[0].legend(loc='center right',  bbox_to_anchor=(1., 0.25), prop={'weight':'bold', 'size': 12})

    ax[0].set_xlabel('Demonstrations')
    ax[0].set_ylabel('Environment Returns')
    # ax[0].tick_params(axis='both', labelsize=18)
    # ax.set_ylim(-10300, 5000)

    ax[1].axhline(y=-14550.8, color='black', linestyle='--', lw=3,
                  label='FLAIR on 10 Demos')
    ax[1].axhline(y=-14835.5, color='green', linestyle='--', lw=3,
                  label='AIRL on 10 Demos')
    ax[1].axhline(y=-11124.2, color='yellow', linestyle='--', lw=3,
                  label='MSRD on 10 Demos')
    ax[1].plot(demonstration, data[:,2],'-o')

    ax[1].set_xlabel('Demonstrations')
    ax[1].set_ylabel('Log Likelihood')
    # ax[1].legend(loc='center right',  bbox_to_anchor=(1., 0.25), prop={'weight':'bold', 'size': 12})

    ax[2].axhline(y=67.2, color='black', linestyle='--', lw=3,
                  label='FLAIR on 10 Demos')
    ax[2].axhline(y=72.0, color='green', linestyle='--', lw=3,
                  label='AIRL on 10 Demos')
    ax[2].axhline(y=70.9, color='yellow', linestyle='--', lw=3,
                  label='MSRD on 10 Demos')
    ax[2].plot(demonstration, data[:,1], '-o')
    # ax[2].legend(loc='center right',  bbox_to_anchor=(1., 0.25), prop={'weight':'bold', 'size': 12})

    ax[2].set_xlabel('Demonstrations')
    ax[2].set_ylabel('Estimated KL Divergence')

    ax[3].axhline(y=0.614, color='black', linestyle='--', lw=3,
                  label='FLAIR on 10 Demos')
    ax[3].axhline(y=0.502, color='green', linestyle='--', lw=3,
                  label='AIRL on 10 Demos')
    ax[3].axhline(y=0.586, color='yellow', linestyle='--', lw=3,
                  label='MSRD on 10 Demos')
    ax[3].plot(demonstration, data[:,4], '-o')
    ax[3].legend(loc='center right',  bbox_to_anchor=(1., 0.8), prop={'weight':'bold', 'size': 12})
    ax[3].set_ylim(0, 1.0)

    ax[3].set_xlabel('Demonstrations')
    ax[3].set_ylabel('Task Reward Correlation')

    fig.tight_layout(pad=2.0)
    fig.suptitle("100 Demonstration Lunar Lander", y=1.0, fontsize=17, weight='bold')
    # plt.plot(range(-1500, 0, 1), range(-1500, 0, 1), color="black", linestyle="--")
    # ax.xaxis.set_tick_params(labelsize=14)
    # ax.yaxis.set_tick_params(labelsize=15)
    # fig.text(0.5, 0.04, 'Ground Truth Reward', ha='center', fontsize=18)
    # fig.text(0.04, 0.5, 'Rescaled Estimated Reward', va='center', rotation='vertical', fontsize=18)
    plt.savefig("data/figs/scaling_experiment_lander_results.png")
    plt.close(fig)
