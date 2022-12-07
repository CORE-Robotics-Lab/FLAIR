import numpy as np
import scipy as sp
import scipy.stats

def rle(inarray):
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.array(inarray)                  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

def split_list_by_lengths(values, lengths):
    """

    >>> split_list_by_lengths([0,0,0,1,1,1,2,2,2], [2,2,5])
    [[0, 0], [0, 1], [1, 1, 2, 2, 2]]
    """
    assert np.sum(lengths) == len(values)
    idxs = np.cumsum(lengths)
    idxs = np.insert(idxs, 0, 0)
    return [ values[idxs[i]:idxs[i+1] ] for i in range(len(idxs)-1)]

def clip_sing(X, clip_val=1):
    U, E, V = np.linalg.svd(X, full_matrices=False)
    E = np.clip(E, -clip_val, clip_val)
    return U.dot(np.diag(E)).dot(V)

def gauss_log_pdf(params, x):
    # assert np.min(x) >= -1
    # assert np.max(x) <= 1
    mean, log_diag_std = params
    # mean, log_diag_std, original_action = params
    N, d = mean.shape
    cov =  np.square(np.exp(log_diag_std))
    diff = x-mean
    # original_action = np.arctanh(x)
    # diff = original_action-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=1)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=1)
    log_probs = norm_term + var_term + exp_term
    # print(np.max(diff), np.min(log_diag_std), np.min(exp_term), np.min(var_term), np.min(log_probs))
    return log_probs #sp.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)
    # tanh_corr = np.sum(np.log(1-np.square(np.tanh(original_action))+1e-6), axis=1)
    # final = log_probs - tanh_corr
    # final = np.clip(final, -50.0, 1000.0)
    # return final

def numerical_integral(params, x):
    return np.log(np.clip((gaussian_pdf(params, x-0.01) + gaussian_pdf(params, x-0.005) + gaussian_pdf(params, x) +
            gaussian_pdf(params, x+0.005) + gaussian_pdf(params, x+0.01))*0.005, 1e-323, None, dtype=np.float64))
    # return np.log((gaussian_pdf(params, x-0.01) + gaussian_pdf(params, x-0.005) + gaussian_pdf(params, x) +
    #         gaussian_pdf(params, x+0.005) + gaussian_pdf(params, x+0.01))*0.005)

def gaussian_pdf(params, x):
    mean, log_diag_std = params
    cov = np.square(np.exp(log_diag_std))
    diff = x-mean
    N, d = mean.shape

    frac_term = (1 / np.sqrt(2*cov*np.pi)) ** d
    exponent_term = -np.square(diff) / (2*cov)

    return frac_term * np.exp(exponent_term)

def numerical_mixture_integral(params, x, weights):
    return np.log(np.clip((gaussian_mixture_pdf(params, x-0.01, weights) + gaussian_mixture_pdf(params, x-0.005, weights) + gaussian_mixture_pdf(params, x, weights) +
            gaussian_mixture_pdf(params, x+0.005, weights) + gaussian_mixture_pdf(params, x+0.01, weights))*0.005, 1e-323, None, dtype=np.float64))

def gaussian_mixture_pdf(params, x, weights):
    mean, log_diag_std = params

    mean = np.einsum('ij,jkm->km', weights, mean, dtype=np.float64)
    log_diag_std = np.ones_like(mean)*-2.7671251
    cov = np.square(np.exp(log_diag_std))

    # cov = np.square(np.exp(log_diag_std))
    # cov = np.einsum('ij,jkm->km', np.square(weights), cov)

    diff = x-mean
    N, d = mean.shape

    frac_term = (1 / np.sqrt(2*cov*np.pi)) ** d
    exponent_term = -np.square(diff) / (2*cov)

    return frac_term * np.exp(exponent_term)

def categorical_log_pdf(params, x, one_hot=True):
    if not one_hot:
        raise NotImplementedError()
    probs = params[0]
    return np.log(np.max(probs * x, axis=1))

