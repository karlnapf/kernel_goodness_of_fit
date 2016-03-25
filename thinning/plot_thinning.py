import numpy as np

__author__ = 'kcx'

arr = np.load('temp_quantiles.npy')


print(np.array_str(arr, precision=2, suppress_small=True))

import matplotlib.pyplot as plt
plt.plot(arr)
plt.legend(['5 %', '10%', '15%'])
plt.xlabel('thinning')
plt.ylabel('type one error')
plt.show()