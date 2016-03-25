import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;

from sgld_test.likelihoods import gen_X, _log_probability

sns.set(color_codes=True)

np.random.seed(1307)

N = 400
X = gen_X(N)
theta1 = np.arange(-2, 2, 0.25)
grid_size = len(theta1)
theta2 = np.arange(-2, 2, 0.25)
theta1, theta2 = np.meshgrid(theta1, theta2)
Z = np.copy(theta1)


for i in range(grid_size):
    for j in range(grid_size):
        probability = _log_probability(theta1[i, j], theta2[i, j], X)
        Z[i, j] = probability

max = np.max(Z)+2

Z = np.exp(Z -max)

print(Z)

plt.figure()
CS = plt.contour(theta1, theta2, Z, 10)
plt.show()
