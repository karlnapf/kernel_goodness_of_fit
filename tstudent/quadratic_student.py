from statsmodels.tsa.stattools import acf
from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn

import numpy as np


def grad_log_normal(x):
    return  -x


N = 500


dfs = range(1, 10, 2)
mc_reps = 100
res = np.empty((0,2))

# for df in dfs:
#
#     for mc in range(mc_reps):
#         print(mc)
#         X = np.random.standard_t(df,N)
#         me = GaussianQuadraticTest(grad_log_normal)
#         U_stat,_ = me.get_statistic_multiple(X)
#         pval = me.compute_pvalue(U_stat)
#         res = np.vstack((res,np.array([df, pval])))
#
# for mc in range(mc_reps):
#
#         X = np.random.randn(N)
#         me = GaussianQuadraticTest(grad_log_normal)
#         U_stat,_ = me.get_statistic_multiple(X)
#         pval = me.compute_pvalue(U_stat)
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
# fig.savefig('../write_up/img/qstudent.pdf')
#
# N = 500
#
#
# dfs = range(1, 10, 2)
# mc_reps = 100
# res = np.empty((0,2))




np.save('results.npy',res)


df = DataFrame(res)
pr =seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()
