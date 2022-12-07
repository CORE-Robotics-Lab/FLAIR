import numpy as np
from inverse_rl.utils.divergence_utils import add_noise, query_tree


def get_rewdiv(env, mixture, policies, demonstration):
    episode_return = 0.0
    dist = 0.0
    for _ in range(10):
        obs = []
        ob = env.reset()
        for timestep in range(1000):
            act = np.dot(mixture, [policy.get_action(ob)[0] for policy in policies])
            act_executed = act
            ob, rew, done, info = env.step(act_executed)
            episode_return += rew
            obs.append(ob)

        expert = demonstration[0]["observations"]
        n, d = expert.shape
        m = 1000
        const = np.log(m) - np.log(n - 1)
        nn = query_tree(expert, expert, 3)
        nnp = query_tree(expert, obs, 3 - 1)
        new_dist = const + d * (np.log(nnp).mean() - np.log(nn).mean())
        dist += new_dist
    return episode_return/10, dist/10