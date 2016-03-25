from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.likelihoods import gen_X
import numpy as np
import matplotlib.pyplot as plt


theta1 = np.arange(-2, 2, 0.025)
theta2 = np.arange(-2, 2, 0.025)

grid_dimension_size = len(theta1)

theta1, theta2 = np.meshgrid(theta1, theta2)

D_theta1 = np.copy(theta1)
D_theta2 = np.copy(theta1)

sample = gen_X(400)

for i in range(grid_dimension_size):
    for j in range(grid_dimension_size):
        th = np.array([theta1[i, j], theta2[i, j]])

        # subsample = np.random.choice(sample, 40)
        stoch_grad_log_lik = np.sum(manual_grad(th[0], th[1], sample), axis=0)  + grad_log_prior(th)

        D_theta1[i, j] = stoch_grad_log_lik[0]
        D_theta2[i, j] = stoch_grad_log_lik[1]

plt.figure()
CS = plt.streamplot(theta1, theta2, D_theta1, D_theta2, density=[0.5, 1])
plt.show()


