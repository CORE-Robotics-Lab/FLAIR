import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.rcParams['axes.labelsize'] = 17
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(1,1, figsize=(6, 5)) #(13,4)

    # ax = dataframe[['AIRL', 'DMSRD']].plot(kind='box', title='Sample Complexity of AIRL vs DMSRD Mixture Optimization', showmeans=True)
    # rectangular box plot

    # 0, 1, 2, 4, 5, 7, 8
    # bplot1 = ax.boxplot([[1538,7230,1490,950,2000,1101,2450], [3600,2890,2870,1516,1620,1148,1495], [2700,9,9,770,240,9,9,2700]],
    #                      vert=True,  # vertical box alignment
    #                      patch_artist=True,  # fill with color
    #                      medianprops=dict(color='red'),
    #                      meanprops={"markerfacecolor":"red", "markeredgecolor":"red"},
    #                      labels=["AIRL", "MSRD", "FLAIR"], showmeans=True)  # will be used to label x-ticks

    # 1, 2, 3, 4, 5, 6, 7, 8
    bplot1 = ax.boxplot([[1000,1500,1900,2300,1900,2740,1300], [1860,1440,1400,6220,2250,3930,904,1229], [503,503,11,7,1506,27,503,6]],
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         medianprops=dict(color='red'),
                         meanprops={"markerfacecolor":"red", "markeredgecolor":"red"},
                         labels=["AIRL", "MSRD", "FLAIR"], showmeans=True)  # will be used to label x-ticks
    # ax.set_title('Sample Complexity of AIRL vs DMSRD Mixture Optimization')
    colors = ['green', 'yellow', 'tab:blue']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    print(bplot1.keys())
    for mean in bplot1['fliers']:
        mean.set_color('red')

    for tick in ax.xaxis.get_major_ticks():
        # tick.label1.set_fontsize(14)
        tick.label1.set_fontweight('bold')
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label1.set_fontsize(14)
        # tick.label1.set_fontweight('bold')

    # lgnd = ax.legen(prop={'weight':'bold'})
    # lgnd.legendHandles[0]._sizes = [30]
    # lgnd.legendHandles[1]._sizes = [30]
    # lgnd.legendHandles[2]._sizes = [30]
    # ax.set_title('DMSRD', fontsize=14)
    ax.set_xlabel('Method', labelpad=5)
    ax.set_ylabel('Episodes', labelpad=10)

    fig.tight_layout(pad=2.0)
    # fig.suptitle("Lunar Lander Sample Complexity", y=1.0, fontsize=17, weight='bold')
    fig.suptitle("Bipedal Walker Sample Complexity", y=1.0, fontsize=17, weight='bold')
    # plt.plot(range(-1500, 0, 1), range(-1500, 0, 1), color="black", linestyle="--")
    # ax.xaxis.set_tick_params(labelsize=14)
    # ax.yaxis.set_tick_params(labelsize=15)
    # fig.text(0.5, 0.04, 'Ground Truth Reward', ha='center', fontsize=18)
    # fig.text(0.04, 0.5, 'Rescaled Estimated Reward', va='center', rotation='vertical', fontsize=18)
    # plt.savefig("data/figs/ll_sample_complexity.png")
    plt.savefig("data/figs/bw_sample_complexity.png")
    plt.close(fig)
