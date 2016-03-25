from scipy.stats import norm
import numpy as np
from sgld_test.constants import SIGMA_x, SIGMA_1, SIGMA_2


def log_probability(theta,x):
    theta_1 = theta[0]
    theta_2 = theta[1]
    return _log_probability(theta_1,theta_2,x)


def _vector_of_log_likelihoods(theta_1, theta_2, x):
    lik = np.log(0.5 * norm.pdf(x, theta_1, SIGMA_x) + 0.5 * norm.pdf(x, theta_1 + theta_2, SIGMA_x))
    return lik


def _log_lik(theta_1, theta_2, x):
    lik = _vector_of_log_likelihoods(theta_1, theta_2, x)
    log_lik = np.sum(lik)
    return log_lik


def _log_probability(theta_1,theta_2,x):
    log_lik = _log_lik(theta_1, theta_2, x)

    log_prior = np.log(norm.pdf(theta_1,0, SIGMA_1)) + np.log(norm.pdf(theta_2,0, SIGMA_2))

    return log_lik+log_prior

def gen_X(n):
    res = []
    true_theta_1 = 0
    true_theta_2 = 1.
    for _ in range(n):
        coin  = np.random.rand()
        if coin < 0.5:
            add = np.random.randn()*SIGMA_x+true_theta_1
        else:
            add = np.random.randn()*SIGMA_x+true_theta_1+true_theta_2

        res.append(add)

    return np.array(res)


