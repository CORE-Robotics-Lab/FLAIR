import tensorflow as tf
from sandbox.rocky.tf.envs.base import TfEnv

from inverse_rl.algos.dmsrd import DMSRD
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.dmsrd import new_likelihood
from inverse_rl.utils.log_utils import rllab_logdir
from global_utils.utils import *
import rllab.misc.logger as logger


def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))

    # load expert demonstrations from DIAYN dataset
    # demonstrations = load_expert_from_core_MSD('data/100mix.pkl', length=1000, repeat_each_skill=3,
    #                                                  separate_styles=True)
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000,  repeat_each_skill=3,
                                                     separate_styles=True)

    dmsrd = DMSRD(env, demonstrations, log_prefix='inverted_pendulum_dmsrd_nomip')
    #dmsrd.rand_idx = [0, 3, 8, 5, 7, 2, 6, 9, 1, 4]
    dmsrd.rand_idx = [3, 7, 5, 6, 0, 8, 1, 9, 4, 2]
    dmsrd.n_epochs = 800

    iteration = 0

    while iteration < len(demonstrations):
        dmsrd.mixture_finding(iteration)

        dmsrd.build_graph(iteration)

        dmsrd.dmsrd_train(iteration)

        iteration += 1


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
