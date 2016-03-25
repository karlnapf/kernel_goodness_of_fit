from scipy.stats import norm
from statsmodels.tsa.stattools import acf

from sampplers.MetropolisHastings import metropolis_hastings
from sampplers.austerity import austerity
from sgld_test.bimodal_SGLD import evSGLD
from sgld_test.constants import SIGMA_1, SIGMA_2
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.likelihoods import gen_X, _vector_of_log_likelihoods, log_probability, _log_lik
from stat_test.linear_time import GaussianSteinTest
from stat_test.quadratic_time import GaussianQuadraticTest, QuadraticMultiple,QuadraticMultiple2

MAGIC_BURNIN_NUMBER = 200

P_CHANGE =0.1

__author__ = 'kcx'
import numpy as np


# np.random.seed(13)
SAMPLE_SIZE = 400
X = gen_X(SAMPLE_SIZE)


def vectorized_log_lik(X,theta):
     return _vector_of_log_likelihoods(theta[0],theta[1],X)

def log_density_prior(theta):
    return np.log(norm.pdf(theta[0],0, SIGMA_1)) + np.log(norm.pdf(theta[1],0, SIGMA_2))

def get_thinning(X,nlags = 50):
    autocorrelation = acf(X, nlags=nlags, fft=True)
    thinning = np.argmin(np.abs(autocorrelation - 0.5)) + 1
    return thinning, autocorrelation

def grad_log_lik(t):
    a = np.sum(manual_grad(t[0],t[1],X),axis=0)  - t[1]/SIGMA_2 -t[0]/SIGMA_1
    return a

pvals = []
no_evals = []
for epsilon in np.linspace(0.001, 0.2,25):
    THINNING_ESTIMAE = 10**4

    sample,evals = austerity(vectorized_log_lik,log_density_prior, X,epsilon,batch_size=50, chain_size=THINNING_ESTIMAE, thinning=1, theta_t=np.random.randn(2))


    thinning, autocorr =  get_thinning(sample[:,0])


    print(' - thinning for epsilon:',thinning,epsilon)

    TEST_SIZE = 500

    e_pvals = []
    e_no_evals = []
    for mc_reps in range(50):
        print(mc_reps)
        sample, evals = austerity(vectorized_log_lik,log_density_prior, X,epsilon,batch_size=50,chain_size=TEST_SIZE + MAGIC_BURNIN_NUMBER, thinning=thinning, theta_t=np.random.randn(2))


        sample = sample[MAGIC_BURNIN_NUMBER:]

        assert sample.shape[0] == TEST_SIZE

        np.save('./data/sample'+str(epsilon), sample)


        me = GaussianQuadraticTest(grad_log_lik)
        qm = QuadraticMultiple2(me)

        p = qm.is_from_null(0.05, sample, 0.1)
        print('evals ', evals)
        print('====     p-value', p)
        # print('====     reject',reject)
        e_no_evals.append(evals)
        e_pvals.append(p)

    no_evals.append(e_no_evals)
    pvals.append(e_pvals)


np.save('no_evals',no_evals)
np.save('pvals',pvals)
