import matplotlib
from pandas import DataFrame
import seaborn
from stat_test.linear_time import GaussianSteinTest

__author__ = 'kcx'
import numpy as np


def grad_log_normal(x):
    return  -x

m=2

N = 250*m


dfs = range(1, 10, 2)
mc_reps = 200
res = np.empty((0,2))

for df in dfs:
    for mc in range(mc_reps):

        X = np.random.standard_t(df,N)
        me = GaussianSteinTest(grad_log_normal,m)
        pvalue = me.compute_pvalue(X)
        res = np.vstack((res,np.array([df, pvalue])))

for mc in range(mc_reps):

        X = np.random.randn(N)
        me = GaussianSteinTest(grad_log_normal,m)
        pvalue = me.compute_pvalue(X)
        res = np.vstack((res,np.array([np.Inf, pvalue])))

# import matplotlib.pyplot as plt
# plt.plot(sorted(res[:,1]))
# plt.show()

np.save('results.npy',res)


df = DataFrame(res)
pr =seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()


fig = pr.get_figure()
fig.savefig('../write_up/img/student.pdf')