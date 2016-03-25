from statsmodels.tsa.stattools import acf
import numpy as np
from sgld_test.mcmc_convergance.cosnt import NUMBER_OF_TESTS, NO_OF_SAMPELS_IN_TEST

__author__ = 'kcx'



samples = np.load('./samples.npy')

arr = acf(samples[1,:,0],nlags=50)


for i in range(1,NUMBER_OF_TESTS*NO_OF_SAMPELS_IN_TEST):
    arr += acf(samples[i,:,0],nlags=50)

arr = arr/(NUMBER_OF_TESTS*NO_OF_SAMPELS_IN_TEST)

print(arr*100.0)
