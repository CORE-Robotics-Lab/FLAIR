import numpy as np
import rllab.misc.logger as logger
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
from datetime import datetime
from global_utils.utils import *
from sandbox.rocky.tf.envs.base import TfEnv
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.algos.airl_state import AIRL
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from inverse_rl.algos.irl_trpo import IRLTRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from inverse_rl.utils.log_utils import rllab_logdir


def main():
    env = TfEnv(CustomGymEnv('CustomInvertedPendulum-v0', record_video=False, record_log=False))
    demonstrations = load_expert_from_core_MSD('data/InvertedPendulum10skillsr15.pkl', length=1000, repeat_each_skill=3,
                                                     separate_styles=True)

    irl_model = AIRL(env=env, expert_trajs=demonstrations[0], state_only=True, fusion=False)
    policy = GaussianMLPPolicy(name=f'policy', env_spec=env.spec,
                               hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=1000,
        batch_size=10000,
        max_path_length=1000,
        discount=0.99,
        store_paths=False,
        discrim_train_itrs=10,
        irl_model_wt=1.0,
        entropy_weight=0,  # from AIRL: this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    now = datetime.now()
    log_path = f"data/inverted_pendulum_airl_single/{now.strftime('%d_%m_%Y_%H_%M_%S')}"
    ground_labels = []
    divergence = []

    auc = roc_auc_score(ground_labels, divergence)

    with rllab_logdir(algo=algo, dirname=log_path + '/auc'):
        logger.record_tabular(f'AUC Score', auc)
        logger.dump_tabular(with_prefix=False, write_header=True)

    # calculate roc curves
    lr_fpr, lr_tpr, thresholds = roc_curve(ground_labels, divergence)
    lowest_iu = None
    ix = None

    for i in range(len(thresholds)):
        ths = thresholds[i]
        above_thresholds = np.array(divergence)
        above_thresholds[above_thresholds > ths] = 0
        above_thresholds[above_thresholds <= ths] = 1

        cm1 = confusion_matrix(ground_labels, above_thresholds)
        sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])

        iu_score = np.abs(auc-sensitivity)+np.abs(auc-specificity)
        if lowest_iu is None or iu_score < lowest_iu:
            lowest_iu = iu_score
            ix = i

        # summarize scores
        with rllab_logdir(algo=algo, dirname=log_path + '/auc'):
            logger.record_tabular(f'Threshold', ths)
            logger.record_tabular(f'Sensitivity', sensitivity)
            logger.record_tabular(f'Specificity', specificity)
            logger.record_tabular(f'IU Score', iu_score)
            logger.record_tabular(f'F1 Score', f1_score(ground_labels, above_thresholds))
            logger.record_tabular(f'ground_labels', ground_labels)
            logger.record_tabular(f'Below', above_thresholds)
            logger.dump_tabular(with_prefix=False, write_header=True)

    pyplot.plot(lr_fpr, lr_tpr, marker='.', lw=2, label=f'ROC Curve (Area = {auc})')
    pyplot.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label='No Skill')
    pyplot.scatter(lr_fpr[ix], lr_tpr[ix], marker='o', color='black', label=f'Best (IU Score = {lowest_iu})')
    # axis ground_labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("DMSRD Receiver Operating Characteristic Curve")
    pyplot.legend(loc="lower right")
    pyplot.savefig(f'{log_path}/roc_curve.png')


""" Standard DMSRD algorithm """
if __name__ == "__main__":
    main()
