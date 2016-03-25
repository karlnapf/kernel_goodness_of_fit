from unittest import TestCase
import numpy as np
from numpy.testing.utils import assert_almost_equal
from sgld_test.bimodal_SGLD import  evSGLD, SGLD
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior

__author__ = 'kcx'


class TestOne_sample_SGLD(TestCase):


    def test_vectorized_SGLD(self):
        np.random.seed(0)
        X = np.arange(5)-2.0
        b=2.31
        a = 0.01584
        epsilons = a*(b+np.arange(5))**(-0.55)


        r1 = evSGLD(manual_grad,grad_log_prior,X,n=1,epsilons=epsilons,theta=np.array([1.,1.3]))
        np.random.seed(0)
        r2 = SGLD(manual_grad,grad_log_prior,X,n=1,chain_size=5,theta=np.array([1.,1.3]))


        assert_almost_equal(r1,r2)
