import os

from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian

import numpy as np
from stat_test.quadratic_time import GaussianQuadraticTest
from tools.tools import store_results


if __name__ == '__main__':
    D = 1
    N_test = 500
    N_fit = 50000
    ms_fit = np.array([1, 2, 5, 10, 25, 50, 75, 100, 250, 500, 1000, 2000, 5000])
    
    sigma = 1
    lmbda = 0.01
    
    grad = lambda x: est.grad(np.array([x]))[0]
    s =  GaussianQuadraticTest(grad)
    num_bootstrap = 200
    
    result_fname = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
    
    num_repetitions = 150
    for _ in range(num_repetitions):
        for m in ms_fit:
            est = KernelExpFiniteGaussian(sigma, lmbda, m, D)
            X_test = np.random.randn(N_test, D)
            
            X = np.random.randn(N_fit, D)
            est.fit(X)
            
            U_matrix, stat = s.get_statistic_multiple(X_test[:,0])
        
            bootsraped_stats = np.empty(num_bootstrap)
            for i in range(num_bootstrap):
                W = np.sign(np.random.randn(N_test))
                WW = np.outer(W, W)
                st = np.mean(U_matrix * WW)
                bootsraped_stats[i] = N_test * st
            
            p_value = np.mean(bootsraped_stats>stat)
            print m, p_value

            store_results(result_fname,
                          D=D,
                          N_fit=N_fit,
                          N_test=N_test,
                          m=m,
                          num_bootstrap=num_bootstrap,
                          sigma=sigma,
                          lmbda=lmbda,
                          p_value=p_value
                          )
