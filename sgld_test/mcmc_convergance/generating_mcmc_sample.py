from time import time
from statsmodels.tsa.stattools import acf
from sgld_test.mcmc_convergance.cosnt import NUMBER_OF_TESTS, NO_OF_SAMPELS_IN_TEST, CHAIN_SIZE, SEED, SAMPLE_SIZE
from sgld_test.likelihoods import gen_X, log_probability

from sampplers.MetropolisHastings import metropolis_hastings
import numpy as np


np.random.seed(SEED)
X = gen_X(SAMPLE_SIZE)

def vectorized_log_density(theta):
     return log_probability(theta,X)

t1 = time()


sample = []
no_chains = NUMBER_OF_TESTS * NO_OF_SAMPELS_IN_TEST
for i in range(no_chains):
    if i % 100 == 0:
        print(i*100.0/no_chains)
        print(time()-t1)
    sample.append(metropolis_hastings(vectorized_log_density, chain_size=CHAIN_SIZE, thinning=1, x_prev=np.random.randn(2)))

sample = np.array(sample)



np.save('samples.npy',sample)

