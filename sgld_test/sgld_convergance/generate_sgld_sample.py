
from time import time
from sgld_test.bimodal_SGLD import vSGLD, evSGLD
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.mcmc_convergance.cosnt import NUMBER_OF_TESTS, NO_OF_SAMPELS_IN_TEST, CHAIN_SIZE, SEED, SGLD_CHAIN_SIZE, \
    SAMPLE_SIZE
from sgld_test.likelihoods import gen_X, log_probability


import numpy as np



np.random.seed(SEED)
X = gen_X(SAMPLE_SIZE)


def vectorized_log_density(theta):
     return log_probability(theta,X)

t1 = time()


sample = []
no_chains = NUMBER_OF_TESTS * NO_OF_SAMPELS_IN_TEST

for i in range(no_chains):
    if i % (100) == 0:
        print(float(i)*100.0/no_chains)
        print(time()-t1)
    sample.append(evSGLD(manual_grad, grad_log_prior, X, n=1, chain_size=SGLD_CHAIN_SIZE,theta = np.random.randn(2) ) )

sample = np.array(sample)

np.save('samples.npy',sample)
