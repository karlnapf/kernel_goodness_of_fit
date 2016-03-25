from time import time
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from sampplers.austerity import austerity
from sgld_test.constants import SIGMA_1, SIGMA_2
from sgld_test.mcmc_convergance.cosnt import NUMBER_OF_TESTS, NO_OF_SAMPELS_IN_TEST, CHAIN_SIZE, SEED, SAMPLE_SIZE
from sgld_test.likelihoods import gen_X, log_probability, _log_lik, _vector_of_log_likelihoods

from sampplers.MetropolisHastings import metropolis_hastings
import numpy as np


np.random.seed(SEED)
X = gen_X(SAMPLE_SIZE)


def vectorized_log_lik(X,theta):
     return _vector_of_log_likelihoods(theta[0],theta[1],X)

def log_density_prior(theta):
    return np.log(norm.pdf(theta[0],0, SIGMA_1)) + np.log(norm.pdf(theta[1],0, SIGMA_2))



sample = austerity(vectorized_log_lik,log_density_prior, X,0.01,batch_size=50,chain_size=20*1000, thinning=1, theta_t=np.random.randn(2))

print(acf(sample[:,1],nlags=50))

#
# import seaborn as sns
# sns.set(color_codes=True)
# with sns.axes_style("white"):
#     pr = sns.jointplot(x=sample[:,0], y=sample[:,1], kind="kde", color="k");
#
#     # pr.savefig('../../write_up/img/mcmc_sample.pdf')
#
#     sns.plt.show()

