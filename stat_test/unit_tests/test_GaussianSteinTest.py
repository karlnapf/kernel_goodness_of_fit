from unittest import TestCase
from stat_test.linear_time import GaussianSteinTest
import numpy as np
__author__ = 'kcx'


class TestMeanEmbeddingTest(TestCase):

    def grad_log_normal(self,x):
        return  -x


    def test_on_one_dim_gaussian(self):
        np.random.seed(42)
        data = np.random.randn(10000)
        me = GaussianSteinTest(self.grad_log_normal,1)
        assert me.compute_pvalue(data)>0.05


    def test_on_four_dim_gaussian(self):
        np.random.seed(42)
        data = np.random.randn(10000,4)

        me = GaussianSteinTest(self.grad_log_normal,1)
        assert me.compute_pvalue(data) > 0.05

    def test_on_one_dim_gaussian_with_three_freqs(self):
        np.random.seed(42)
        data = np.random.randn(10000)
        me = GaussianSteinTest(self.grad_log_normal,3)
        assert me.compute_pvalue(data)>0.05


    def test_on_four_dim_gaussian_with_three_freqs(self):
        np.random.seed(42)
        data = np.random.randn(10000,4)
        me = GaussianSteinTest(self.grad_log_normal,3)
        assert me.compute_pvalue(data) > 0.05


    def test_power_growth(self):
        np.random.seed(42)
        data = np.random.randn(10000,4)+0.01*np.random.rand()
        me = GaussianSteinTest(self.grad_log_normal,10)
        p1 = me.compute_pvalue(data)
        me = GaussianSteinTest(self.grad_log_normal,1)
        p2 = me.compute_pvalue(data)
        assert p1 <p2
