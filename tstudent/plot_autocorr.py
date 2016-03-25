from sampplers.MetropolisHastings import metropolis_hastings

__author__ = 'kcx'
from statsmodels.tsa.stattools import acf
from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn
import numpy as np
from tools.latex_plot_init import plt

SGLD_EPSILON = 0.0478

P_CHANGE = 0.1

N = 500

DEGREES_OF_FREEDOM = [1,3,5,7,9,11]+[1000]
MC_PVALUES_REPS = 400
TEST_CHAIN_SIZE = 2*10**6


# The null statistic is that random variables come form normal distibution, so the test statistic takes a gradient of
# logarithm of density of standard normal.
def grad_log_normal(x):
    return -x

def log_normal(x):
    return -x**2/2

def grad_log_t_df(df):
    def grad_log_t(x):
        return -(df+1.0)/2.0*np.log(1+x**2/df)
    return grad_log_t

def sample_sgld_t_student(N,degree_of_freedom,epsilon):
    grd_log = grad_log_t_df(degree_of_freedom)
    X =  metropolis_hastings(grd_log, chain_size=N, thinning=1, x_prev=np.random.randn(),step=0.50)
    return X



# estimate size of thinning
def get_thinning(X,nlags = 50):
    autocorrelation = acf(X, nlags=nlags, fft=True)
    # find correlation closest to given v
    thinning = np.argmin(np.abs(autocorrelation - 0.5)) + 1
    return thinning, autocorrelation

def normal_mild_corr(N):
    X =  metropolis_hastings(log_normal, chain_size=N, thinning=1, x_prev=np.random.randn(),step=0.55)
    return X


X = normal_mild_corr(TEST_CHAIN_SIZE)
sgld_thinning, autocorr = get_thinning(X,500)
print('thinning for sgld t-student simulation ', sgld_thinning,autocorr[sgld_thinning])


X = normal_mild_corr(sgld_thinning *100000)
X = X[::sgld_thinning]

r= acf(X,nlags=30)
print(r)

seaborn.set_style("whitegrid")
plt.plot(r)
plt.xlabel('lags')
plt.ylabel('auto correlation')
plt.ylim([0,1])
plt.tight_layout()
plt.savefig('../write_up/img/sgld_lags.eps')