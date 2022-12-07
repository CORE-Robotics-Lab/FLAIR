from sandbox.rocky.tf.envs.base import TfEnv

from inverse_rl.algos.dmsrd_algo import DMSRD
from inverse_rl.envs.env_utils import CustomGymEnv
from global_utils.utils import *
import csv


"""
Dynamic Multi Strategy Reward Distillation on Inverted Pendulum
"""
def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000, repeat_each_skill=3, separate_styles=True)

    dmsrd = DMSRD(env, demonstrations, log_prefix='inverted_pendulum_dmsrd',
                                               airl_itrs=600, msrd_itrs=400, bool_save_vid=False)
    dmsrd.rand_idx = [3, 7, 5, 6, 0, 8, 1, 9, 4, 2]

    for iteration in range(len(demonstrations)):
        dmsrd.new_demonstration(iteration)
        dmsrd.mixture_optimize(iteration)
        dmsrd.msrd(iteration)

    with open(f'{dmsrd.log_path}/mixtures.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerows([["Order of Demonstrations"]])
        writer.writerow(dmsrd.rand_idx)
        writer.writerows([["Mixtures"]])
        writer.writerows(dmsrd.mixture_weights)


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
