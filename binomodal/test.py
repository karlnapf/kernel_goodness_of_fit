from stat_test.linear_time import GaussianSteinTest

__author__ = 'kcx'
import numpy as np



def grad_log_normal(m):
    def grad_log_mix(x):
        e2mx = np.exp(2 * m * x)
        nom = m - e2mx * m + x + e2mx * x
        denom = 1 + e2mx
        return -nom / denom

    return grad_log_mix

m=5

me = GaussianSteinTest(grad_log_normal(m),m)

X = np.random.randn(10000)-m

print(me.compute_pvalue(X))