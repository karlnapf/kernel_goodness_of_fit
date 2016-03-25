from unittest import TestCase
from time import time
from numpy.testing.utils import assert_almost_equal
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from sampplers.austerity import austerity
from sgld_test.constants import SIGMA_1, SIGMA_2
from sgld_test.mcmc_convergance.cosnt import NUMBER_OF_TESTS, NO_OF_SAMPELS_IN_TEST, CHAIN_SIZE, SEED, SAMPLE_SIZE
from sgld_test.likelihoods import gen_X, log_probability, _log_lik, _vector_of_log_likelihoods
import numpy as np


__author__ = 'kcx'


class TestAusterity(TestCase):

    def test_austerity(self):
        np.random.seed(SEED)
        X = gen_X(SAMPLE_SIZE)


        def vectorized_log_lik(X,theta):
            return _vector_of_log_likelihoods(theta[0],theta[1],X)

        def log_density_prior(theta):
            return np.log(norm.pdf(theta[0],0, SIGMA_1)) + np.log(norm.pdf(theta[1],0, SIGMA_2))


        sample,_ = austerity(vectorized_log_lik,log_density_prior, X,0.01,batch_size=50,chain_size=10, thinning=1, theta_t=np.random.randn(2))
        assert_almost_equal(np.array([-0.2554517,  1.3805683]),sample[-1])
