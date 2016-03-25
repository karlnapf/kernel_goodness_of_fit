
from unittest import TestCase

import numpy as np

from numpy.testing.utils import assert_allclose, assert_almost_equal

from goodness_of_fit.test import _GoodnessOfFitTest,GoodnessOfFitTest


__author__ = 'kcx, heiko'


class TestGaussianQuadraticTest(TestCase):
    def grad_log_normal(self, x):
        return -x

    def  test_corr(self):
        np.random.seed(43)
        sigma = np.array([[1,0.5],[0.5,1]])
        def grad_log_correleted(x):
            sigmaInv = np.linalg.inv(sigma)
            return - np.dot(sigmaInv.T + sigmaInv, x)/2.0

        me = GoodnessOfFitTest(grad_log_correleted)

        X =  np.random.multivariate_normal([0,0], sigma, 200)


        p_val = me.is_from_null(0.05, X, 0.1)
        np.testing.assert_almost_equal(0.235,p_val)


    def  test_corr2(self):
        np.random.seed(43)
        sigma = np.array([[1,0.5],[0.5,1]])
        def grad_log_correleted(x):
            sigmaInv = np.linalg.inv(sigma)
            return - np.dot(sigmaInv.T + sigmaInv, x)/10.0

        me = GoodnessOfFitTest(grad_log_correleted)

        X =  np.random.multivariate_normal([0,0], sigma, 200)

        p_val = me.is_from_null(0.05, X, 0.1)
        np.testing.assert_almost_equal(0,p_val)

