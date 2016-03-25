from sampplers.MetropolisHastings import metropolis_hastings

__author__ = 'kcx'
from statsmodels.tsa.stattools import acf
from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn
import numpy as np
from tools.latex_plot_init import plt

import itertools


N = 1400

DEGREES_OF_FREEDOM = [1, 3,6,9,11,np.Inf]
MC_PVALUES_REPS = 100
TEST_CHAIN_SIZE = 2 * 10 ** 6


# The null statistic is that random variables come form normal distibution, so the test statistic takes a gradient of
# logarithm of density of standard normal.
def grad_log_normal(x):
    return -x


def log_normal(x):
    return -x ** 2.0 / 2.0


def grad_log_t_df(df):
    def grad_log_t(x):
        return -(df + 1.0) / 2.0 * np.log(1 + x ** 2 / df)

    return grad_log_t


def gen(N, df, thinning=1):
    log_den = log_normal
    if df < np.Inf:
        log_den = grad_log_t_df(df)

    return metropolis_hastings(log_den, chain_size=N, thinning=thinning, x_prev=np.random.randn(), step=0.5)


# estimate size of thinning
def get_thinning(X, nlags=50):
    autocorrelation = acf(X, nlags=nlags, fft=True)
    thinning = np.argmin(np.abs(autocorrelation - 0.95)) + 1
    return thinning, autocorrelation

#
# X = gen(TEST_CHAIN_SIZE, np.Inf)
# thinning, autocorr = get_thinning(X)
# print('thinning for AR normal simulation ', thinning, autocorr[thinning])

thinning = 1
tester = GaussianQuadraticTest(grad_log_normal)


def get_pval(X, tester, p_change):
    U_stat, _ = tester.get_statistic_multiple(X)
    return tester.compute_pvalues_for_processes(U_stat, p_change)


def get_pair(sample_size, df, thinning, tester, p_change):
    X = gen(sample_size, df, thinning)
    pval = get_pval(X, tester, p_change)
    return [df, pval]

P_CHANGE = 0.1
results = []
thinning = 20
print('===best')
for df in DEGREES_OF_FREEDOM*MC_PVALUES_REPS:
    print(df)
    pair = get_pair(N , df, thinning, tester, P_CHANGE)
    results.append(pair)

np.save('results_thinning.npy', results)


P_CHANGE = 0.5
results = []
thinning=1
print('===bad')
for df in DEGREES_OF_FREEDOM*MC_PVALUES_REPS:
    print(df)
    pair = get_pair(N , df, thinning, tester, P_CHANGE)
    results.append(pair)

np.save('results_bad.npy', results)

print('===good')
P_CHANGE = 0.02
results = []
for df in DEGREES_OF_FREEDOM*MC_PVALUES_REPS:
    print(df)
    pair = get_pair(N , df, thinning, tester, P_CHANGE)
    results.append(pair)

np.save('results_good.npy', results)


