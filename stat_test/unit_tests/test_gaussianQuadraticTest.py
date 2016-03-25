from nose.tools import assert_almost_equal
from unittest import TestCase

import numpy as np
from stat_test.quadratic_time import GaussianQuadraticTest, QuadraticMultiple
from numpy.testing.utils import assert_allclose


__author__ = 'kcx, heiko'


class TestGaussianQuadraticTest(TestCase):
    def grad_log_normal(self, x):
        return -x

    def test_two_dimensional_tests_null(self):
        np.random.seed(43)
        me = GaussianQuadraticTest(self.grad_log_normal)
        samples = np.random.randn(100,2)
        U,_ = me.get_statisitc_two_dim(100,samples,1)
        p = me.compute_pvalue(U)
        assert p == 0.63


    def test_two_dimensional_tests_alt(self):
        np.random.seed(43)
        me = GaussianQuadraticTest(self.grad_log_normal)
        samples = np.random.randn(100,2)+1
        U,_ = me.get_statisitc_two_dim(100,samples,1)
        p = me.compute_pvalue(U)
        assert p == 0

    def  test_corr(self):
        np.random.seed(43)
        sigma = np.array([[1,0.5],[0.5,1]])
        def grad_log_correleted(x):
            sigmaInv = np.linalg.inv(sigma)
            return - np.dot(sigmaInv.T + sigmaInv, x)/2.0

        me = GaussianQuadraticTest(grad_log_correleted)
        qm = QuadraticMultiple(me)
        X =  np.random.multivariate_normal([0,0], sigma, 200)


        reject,p_val = qm.is_from_null(0.05, X, 0.1)
        np.testing.assert_almost_equal([0.465,  0.465],p_val)


    def test_two_dimensional_tests_agrees(self):
        np.random.seed(43)
        me = GaussianQuadraticTest(self.grad_log_normal)
        samples = np.random.randn(10,2)
        U1,_ = me.get_statisitc_two_dim(10,samples,1)
        U2,_ = me.get_statistic_multiple_dim(samples,1)
        np.testing.assert_almost_equal(U1,U2)



    def test_regression_1(self):
        np.random.seed(43)
        data = np.random.randn(100)
        me = GaussianQuadraticTest(self.grad_log_normal)
        U_stat,_ = me.get_statistic_multiple(data)
        pval = me.compute_pvalue(U_stat)
        assert pval == 0.79

    def test_regression_2(self):
        np.random.seed(42)
        data = np.random.randn(100) * 2.0
        me = GaussianQuadraticTest(self.grad_log_normal)
        U_stat,_ = me.get_statistic_multiple(data)
        pval = me.compute_pvalue(U_stat)
        assert pval == 0.0

    def test_k_multiple_equals_k_no_grad_multiple_given(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        K = me.k_multiple(X)
        
        for i in range(N):
            for j in range(N):
                k = me.k(X[i], X[j])
                assert_almost_equal(K[i, j], k)

    def test_k_multiple_equals_k_no_dim(self):
        N = 10
        X = np.random.randn(N,1)
        me = GaussianQuadraticTest(self.grad_log_normal)
        K1 = me.k_multiple_dim(X)
        K2  =me.k_multiple(X[:,0])
        np.testing.assert_almost_equal(K1, K2)

    def test_g1k_multiple_dim(self):
        N = 10
        X = np.random.randn(N,1)
        me = GaussianQuadraticTest(self.grad_log_normal)
        K = me.k_multiple_dim(X)
        g1k_alt = me.g1k_multiple_dim(X,K,0)
        g1k_orig = me.g1k_multiple(X[:,0])
        np.testing.assert_almost_equal(g1k_alt, g1k_orig)


    def test_gk_multiple_dim(self):
        N  = 10
        X  = np.random.randn(N,1)
        me = GaussianQuadraticTest(self.grad_log_normal)
        K  = me.k_multiple_dim(X)
        gk_alt  = me.gk_multiple_dim(X,K,0)
        gk_orig = me.gk_multiple(X[:,0])
        np.testing.assert_almost_equal(gk_alt, gk_orig)

    def test_k_multiple_equals_k_grad_multiple_given(self):
        def fun(self, X):
            return -X
        
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal, grad_log_prob_multiple=fun)
        K = me.k_multiple(X)
        
        for i in range(N):
            for j in range(N):
                k = me.k(X[i], X[j])
                assert_almost_equal(K[i, j], k)

    def test_g1k_multiple_equals_g1k(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        G1K = me.g1k_multiple(X)
         
        for i in range(N):
            for j in range(N):
                g1k = me.g1k(X[i], X[j])
                assert_almost_equal(G1K[i, j], g1k)
    
    def test_g2k_multiple_equals_g2k(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        G2K = me.g2k_multiple(X)
         
        for i in range(N):
            for j in range(N):
                g2k = me.g2k(X[i], X[j])
                assert_almost_equal(G2K[i, j], g2k)

    def test_gk_multiple_equals_gk(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        GK = me.gk_multiple(X)
         
        for i in range(N):
            for j in range(N):
                gk = me.gk(X[i], X[j])
                assert_almost_equal(GK[i, j], gk)
    
    def test_get_statistic_multiple_equals_get_statistic(self):
        N = 10
        X = np.random.randn(N)
        me = GaussianQuadraticTest(self.grad_log_normal)
        U_matrix_multiple, stat_multiple = me.get_statistic_multiple(X)
        U_matrix, stat = me.get_statisitc(N, X)
        
        assert_allclose(stat, stat_multiple)
        assert_allclose(U_matrix_multiple, U_matrix)