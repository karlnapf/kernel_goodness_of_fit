from statsmodels.tsa.stattools import acf

__author__ = 'kcx'

from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn

import numpy as np


def grad_log_normal(x):
    return  -x


N = 500


dfs = range(1, 11, 2)
mc_reps = 100
res = np.empty((0,2))

# for df in dfs:
#
#     for mc in range(mc_reps):
#         print(mc)
#         X = np.random.standard_t(df,N)
#         me = GaussianQuadraticTest(grad_log_normal)
#         U_stat,_ = me.get_statistic_multiple(X)
#         pval = me.compute_pvalues_for_processes(U_stat,0.5)
#         res = np.vstack((res,np.array([df, pval])))
#
# for mc in range(mc_reps):
#
#         X = np.random.randn(N)
#         me = GaussianQuadraticTest(grad_log_normal)
#         U_stat,_ = me.get_statistic_multiple(X)
#         pval = me.compute_pvalues_for_processes(U_stat,0.5)
#         res = np.vstack((res,np.array([np.Inf, pval])))
#
#
# np.save('results.npy',res)
#
#
# df = DataFrame(res)
# pr =seaborn.boxplot(x=0,y=1,data=df)
# seaborn.plt.show()
#
#
# fig = pr.get_figure()
# fig.savefig('../write_up/img/pqstudent.pdf')



def correlatet_t(X,N):

    fc = np.random.rand(N)
    # for i in range(1,N):
    #     if fc[i]>0.05:
    #         X[i] = X[i-1]
    X= X + np.random.randn(N)*0.01
    return X

def almost_t_student(N,df,epsilon):
    samples = np.zeros(N)
    xt = 0
    for t in range(N):
        delta = epsilon/2.0*(-(1+df)*xt/(df+xt**2.0)) + np.sqrt(epsilon)*np.random.randn()
        xt = xt + delta
        samples[t] = xt
    return samples

#
X = almost_t_student(20000,50.0,0.01)

import seaborn as sns
sns.set(color_codes=True)
sns.distplot(X);
sns.plt.show()
print(acf(X,nlags=10))

dfs = range(1, 4, 2)
mc_reps = 100
res = np.empty((0,2))

block = N/np.log(N)
p_change  = 1.0/block
print(p_change)

for df in dfs:

    for mc in range(mc_reps):
        print(mc)
        X = almost_t_student(10*N,df,0.01)
        X = X[::10]
        me = GaussianQuadraticTest(grad_log_normal)
        U_stat,_ = me.get_statistic_multiple(X)

        pval = me.compute_pvalues_for_processes(U_stat,p_change)
        res = np.vstack((res,np.array([df, pval])))

for mc in range(mc_reps):
        X = almost_t_student(10*N,100,0.01)
        X = X[::10]
        me = GaussianQuadraticTest(grad_log_normal)
        U_stat,_ = me.get_statistic_multiple(X)
        pval = me.compute_pvalues_for_processes(U_stat,p_change)
        res = np.vstack((res,np.array([np.Inf, pval])))

np.save('results.npy',res)

df = DataFrame(res)
pr =seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()
