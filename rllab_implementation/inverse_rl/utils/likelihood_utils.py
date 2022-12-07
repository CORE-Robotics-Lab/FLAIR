import numpy as np


# Calculate the mixture likelihood given weights and probabilities
def grid_objective(weights, action_prob):
    out = np.matmul(weights, action_prob, dtype=np.float64)
    out = np.sum(safe_log_np(out), axis=1, dtype=np.float64)
    return out


# Calculate a safe numpy log
def safe_log_np(x):
    x = np.clip(x, 1e-37, None, dtype=np.float64)
    return np.log(x, dtype=np.float64)


# Perform clipping on log calculation
def new_likelihood(action_prob):
    out = np.exp(action_prob, dtype=np.float64)
    return np.sum(safe_log_np(out), dtype=np.float64)


# Perform grid search to find best mixture weights to optimize likelihood
def Grid_Search(action_prob, shots):
    def objective(weights, action_prob):
        out = np.matmul(weights, action_prob, dtype=np.float64)
        out = np.sum(safe_log_np(out), axis=1, dtype=np.float64)
        return out

    # action_prob = np.clip(action_prob, None, 0.)
    action_prob = np.exp(action_prob, dtype=np.float64)
    action_prob = np.resize(action_prob, (action_prob.shape[0], action_prob.shape[1]*action_prob.shape[2]))

    num_pols = action_prob.shape[0]

    weights = np.random.uniform(0, 1, (shots, num_pols))
    weights = np.append(weights, np.diag(np.arange(num_pols)), axis=0)
    for i in range(len(weights)):
        if np.sum(weights[i]) <= 0:
            weights[i] = np.ones_like(weights[i], dtype=np.float64)
        weights[i] /= np.sum(weights[i], dtype=np.float64)

    F = objective(weights, action_prob)

    best_idx = np.argmax(F)
    best_likelihood = F[best_idx]
    best_mix = weights[best_idx]

    return best_mix, best_likelihood


# Perform grid search to find best mixture weights to optimize gaussian mixture likelihood
def Gaussian_Sum_Likelihood(policies, reward_f, demonstration, shots):
    num_pols = len(policies)
    weights = np.random.uniform(0, 1, (shots, num_pols))
    weights = np.append(weights, np.diag(np.arange(num_pols)), axis=0)

    best_logprob = None
    best_mix = None
    best_logprobs_traj = None
    for i in range(len(weights)):
        if np.sum(weights[i]) <= 0:
            weights[i] = np.ones_like(weights[i], dtype=np.float64)
        weights[i] /= np.sum(weights[i], dtype=np.float64)
        logprobs_traj = np.array(reward_f.eval_expert_mix(demonstration, policies, weights[i]), dtype=np.float64)
        logprob = new_likelihood(logprobs_traj)
        if best_logprob is None or logprob > best_logprob:
            best_mix = weights[i]
            best_logprob = logprob
            best_logprobs_traj = logprobs_traj

    return best_mix, best_logprob, best_logprobs_traj
