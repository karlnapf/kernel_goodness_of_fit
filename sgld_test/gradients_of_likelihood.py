from sgld_test.constants import SIGMA_x, SIGMA_1, SIGMA_2

__author__ = 'kcx'
import numpy as np

def grad_log_prior(theta):
    return -theta/[SIGMA_1,SIGMA_2]


def lik_2(theta_1, theta_2, x):
    arg = (x - theta_1 - theta_2)
    return 1.0 / np.sqrt(2 * SIGMA_x ** 2 * np.pi) * np.exp(- arg ** 2 / (2 * SIGMA_x ** 2))


def lik_1(theta_1, theta_2, x):
    arg = (x - theta_1)
    return 1.0 / np.sqrt(2 * SIGMA_x ** 2 * np.pi) * np.exp(-  arg ** 2 / (2 * SIGMA_x ** 2))



def manual_grad(theta1, theta2, x):
    lik_theta_1 = lik_1(theta1, theta2, x)
    lik_theta_2 = lik_2(theta1, theta2, x)
    lik_mixture = (lik_theta_1 + lik_theta_2) / 2.0

    twoSigmaSquare = (2 * SIGMA_x ** 2)

    derivative_inside_log_wrt_theta_1 = lik_theta_1 * (-theta1 + x) / twoSigmaSquare
    derivative_inside_log_wrt_theta_2 = lik_theta_2 * (-theta1 - theta2 + x) / twoSigmaSquare

    d_theta_2 = (derivative_inside_log_wrt_theta_1 + derivative_inside_log_wrt_theta_2) / lik_mixture
    d_theta_1 = derivative_inside_log_wrt_theta_2 / lik_mixture
    return np.array([d_theta_2, d_theta_1]).T


