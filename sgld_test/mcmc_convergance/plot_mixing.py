from pandas import DataFrame
import seaborn
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.mcmc_convergance.cosnt import CHAIN_SIZE, NUMBER_OF_TESTS, NO_OF_SAMPELS_IN_TEST, SEED, SAMPLE_SIZE
from sgld_test.likelihoods import gen_X, log_probability
from stat_test.linear_time import GaussianSteinTest
from stat_test.quadratic_time import GaussianQuadraticTest, QuadraticMultiple

__author__ = 'kcx'
import  numpy as np

samples = np.load('./samples.npy')

np.random.seed(SEED)
X = gen_X(SAMPLE_SIZE)

def grad_log_pob(theta):
    s=[]
    for t in theta:
        s.append( np.sum(manual_grad(t[0],t[1],X),axis=0))
    return np.array(s)

me = GaussianSteinTest(grad_log_pob,1)

times_we_look_at = range(0,CHAIN_SIZE,1)
# arr = np.empty((0,2))
#
#
# for time in times_we_look_at:
#     chain_at_time = samples[:,time]
#     print(time)
#     list_of_chain_slices = np.split(chain_at_time,NUMBER_OF_TESTS)
#     for chains_slice in list_of_chain_slices:
#         assert chains_slice.shape == (NO_OF_SAMPELS_IN_TEST,2)
#         pval = me.compute_pvalue(chains_slice)
#         arr = np.vstack((arr, np.array([time,pval])))
#
#
#
# df = DataFrame(arr)
#
# pr = seaborn.boxplot(x=0,y=1,data=df)
# seaborn.plt.show()
# fig = pr.get_figure()
# fig.savefig('../../write_up/img/mcmc_mixing.pdf')

arr = []


me = GaussianSteinTest(grad_log_pob,1)

for time in times_we_look_at:
    chain_at_time = samples[:,time]
    # print(time)
    # pval = me.compute_pvalue(chain_at_time)
    # arr.append(pval)
    def grad_log_pob(t):
        a = np.sum(manual_grad(t[0],t[1],X),axis=0) + grad_log_prior(t)
        return a


    P_CHANGE =0.1

    me = GaussianQuadraticTest(grad_log_pob)
    qm = QuadraticMultiple(me)



    reject, p = qm.is_from_null(0.05, chain_at_time, 0.1)


    print(reject)

# import matplotlib.pyplot as plt
#
# print(arr)
#
# plt.plot(arr)
#
# plt.show()