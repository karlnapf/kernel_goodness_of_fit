import numpy as np

def metropolis_hastings(log_density, chain_size=10000, thinning=15, x_prev=np.random.randn(),step=1):
    A = [x_prev]

    dimension=1
    if hasattr(x_prev, "__len__"):
        dimension = len(x_prev)


    old_log_lik = log_density(x_prev)
    for i in range(chain_size*thinning-1):
        guess = step*np.random.randn(dimension)+x_prev
        new_log_lik = log_density(guess)
        if new_log_lik > old_log_lik:
            A.append(guess)
            old_log_lik = new_log_lik
        else:
            u = np.random.uniform(0.0,1.0)
            if u < np.exp(new_log_lik - old_log_lik):
                A.append(guess)
                old_log_lik = new_log_lik
            else:
                A.append(x_prev)
        x_prev = A[-1]

    array = np.array(A[::thinning])

    return array




class mh_generator:

    def __init__(self,log_density,x_start=np.random.randn()):
        self.log_density = log_density
        self.x_last = x_start

    def get(self, chunk_size,thinning):
        data = metropolis_hastings(self.log_density,chain_size=chunk_size,thinning=thinning,x_prev=self.x_last)
        self.x_last = data[-1]
        return data