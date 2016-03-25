import numpy as np


#   THIS IS ULTRA SPECIFIC TO THE  PROBLEM, Dont dare to use it!!!!
TRUE_B = 2.3101


def SGLD(grad_log_density,grad_log_prior, X,n,chain_size=10000, thinning=1, theta=np.random.rand(2)):

    N = X.shape[0]
    X = np.random.permutation(X)

    samples = np.zeros((chain_size,2))
    for t in range(chain_size*thinning):

        b=2.31
        a = 0.01584
        epsilon_t = a*(b+t)**(-0.55)

        noise = np.sqrt(epsilon_t)*np.random.randn(2)

        sub = np.random.choice(X, n)

        stupid_sum=np.array([0.0,0.0])
        for data_point in sub:
            stupid_sum = stupid_sum+ grad_log_density(theta[0], theta[1],data_point)

        grad = grad_log_prior(theta) + (N/n)*stupid_sum

        grad = grad*epsilon_t/2


        theta = theta+grad+noise

        samples[t] = theta


    return np.array(samples[::thinning])




# b=2.31
#         a = 0.01584
#         epsilon_t = a*(b+t)**(-0.55)
#         epsilon_t = np.max(min_epsilon,epsilon_t)


def evSGLD(grad_log_density,grad_log_prior, X,n,epsilons, theta=None,dim=2):
    if theta is None:
        theta=np.random.randn(dim)
    N = X.shape[0]
    X = np.random.permutation(X)

    chain_size = len(epsilons)
    samples = np.zeros((chain_size,dim))
    for t in range(chain_size):


        noise = np.sqrt(epsilons[t])*np.random.randn(dim)

        sub = np.random.choice(X, n)

        stupid_sum = np.sum(grad_log_density(theta[0],theta[1],sub),axis=0)

        grad = grad_log_prior(theta) + (N/n)*stupid_sum

        grad = grad*epsilons[t]/2

        theta = theta+grad+noise
        samples[t] = theta

    return np.array(samples)