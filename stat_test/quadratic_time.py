from scipy.spatial.distance import squareform, pdist

import numpy as np
from statsmodels.stats.multitest import multipletests
from stat_test.ar import simulate, simulatepm


__author__ = 'kcx, heiko'


class GaussianQuadraticTest:
    def __init__(self, grad_log_prob, scaling=2.0, grad_log_prob_multiple=None):
        self.scaling = scaling
        self.grad = grad_log_prob
        
        # construct (slow) multiple gradient handle if efficient one is not given
        if grad_log_prob_multiple is None:
            def grad_multiple(X):
                # simply loop over grad calls. Slow
                return np.array([self.grad(x) for x in X])
            
            self.grad_multiple = grad_multiple
        else:
            self.grad_multiple = grad_log_prob_multiple
            
    def k(self, x, y):
        return np.exp(-np.dot(x - y,x - y) / self.scaling)
    
    def k_multiple(self, X):
        """
        Efficient computation of kernel matrix without loops
        
        Effectively does the same as calling self.k on all pairs of the input
        """
        assert(X.ndim == 1)
        
        sq_dists = squareform(pdist(X.reshape(len(X), 1), 'sqeuclidean'))
            
        K = np.exp(-(sq_dists) / self.scaling)
        return K

    def k_multiple_dim(self, X):

        # check for stupid mistake
        assert X.shape[0] > X.shape[1]

        sq_dists = squareform(pdist(X, 'sqeuclidean'))

        K = np.exp(-(sq_dists) / self.scaling)
        return K


    def g1k(self, x, y):
        return -2.0 / self.scaling * self.k(x, y) * (x - y)
    
    def g1k_multiple(self, X):
        """
        Efficient gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.g1k on all pairs of the input
        """
        assert X.ndim == 1
        
        differences = X.reshape(len(X), 1) - X.reshape(1, len(X))
        sq_differences = differences ** 2
        K = np.exp(-sq_differences / self.scaling)

        return -2.0 / self.scaling * K * differences


    def g1k_multiple_dim(self, X,K,dim):

        X_dim = X[:,dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1,len(X_dim))

        return -2.0 / self.scaling * K * differences




    def g2k(self, x, y):
        return -self.g1k(x, y)
    
    def g2k_multiple(self, X):
        """
        Efficient 2nd gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.g2k on all pairs of the input
        """
        return -self.g1k_multiple(X)

    def g2k_multiple_dim(self, X,K,dim):
        return -self.g1k_multiple_dim(X,K,dim)

    def gk(self, x, y):
        return 2.0 * self.k(x, y) * (self.scaling - 2 * (x - y) ** 2) / self.scaling ** 2

    def gk_multiple(self, X):
        """
        Efficient gradient computation of Gaussian kernel with multiple inputs
        
        Effectively does the same as calling self.gk on all pairs of the input
        """
        assert X.ndim == 1
        
        differences = X.reshape(len(X), 1) - X.reshape(1, len(X))
        sq_differences = differences ** 2
        K = np.exp(-sq_differences / self.scaling)

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2

    def gk_multiple_dim(self, X,K,dim):
        X_dim = X[:,dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1,len(X_dim))

        sq_differences = differences ** 2

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2


    def get_statisitc(self, N, samples):
        U_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                x1 = samples[i]
                x2 = samples[j]
                a = self.grad(x1) * self.grad(x2) * self.k(x1, x2)
                b = self.grad(x2) * self.g1k(x1, x2)
                c = self.grad(x1) * self.g2k(x1, x2)
                d = self.gk(x1, x2)
                U_matrix[i, j] = a + b + c + d
        stat = N * np.mean(U_matrix)
        return U_matrix, stat


    def get_statisitc_two_dim(self, N, samples,dim):
        U_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                x1 = samples[i]
                x2 = samples[j]
                a = self.grad(x1)[dim] * self.grad(x2)[dim] * self.k(x1, x2)
                b = self.grad(x2)[dim] * self.g1k(x1, x2)[dim]
                c = self.grad(x1)[dim] * self.g2k(x1, x2)[dim]
                d = self.gk(x1, x2)[dim]
                U_matrix[i, j] = a + b + c + d
        stat = N * np.mean(U_matrix)
        return U_matrix, stat



    def get_statistic_multiple_dim(self, samples,dim):

        log_pdf_gradients = self.grad_multiple(samples)
        log_pdf_gradients = log_pdf_gradients[:,dim]
        K = self.k_multiple_dim(samples)
        G1K = self.g1k_multiple_dim(samples,K,dim)
        G2K = self.g2k_multiple_dim(samples,K,dim)
        GK = self.gk_multiple_dim(samples,K,dim)

        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U)
        return U, stat


    def get_statistic_multiple(self, samples):
        """
        Efficient statistic computation with multiple inputs
        
        Effectively does the same as calling self.get_statisitc.
        """
        log_pdf_gradients = self.grad_multiple(samples)
        K = self.k_multiple(samples)
        G1K = self.g1k_multiple(samples)
        G2K = self.g2k_multiple(samples)
        GK = self.gk_multiple(samples)
        
        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U) 
        return U, stat

    def get_statistic_multiple_custom_gradient(self, samples, log_pdf_gradients):
        """
        Implements the statistic for multiple samples, each from a different
        density whose gradient at the sample is passed
        
        """
        K = self.k_multiple(samples)
        G1K = self.g1k_multiple(samples)
        G2K = self.g2k_multiple(samples)
        GK = self.gk_multiple(samples)
        
        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U) 
        return U, stat

    def compute_pvalue(self, U_matrix, num_bootstrapped_stats=100):
        N = U_matrix.shape[0]
        bootsraped_stats = np.zeros(num_bootstrapped_stats)

        for proc in range(num_bootstrapped_stats):
            W = np.sign(np.random.randn(N))
            WW = np.outer(W, W)
            st = np.mean(U_matrix * WW)
            bootsraped_stats[proc] = N * st

        stat = N*np.mean(U_matrix)

        return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats

    def compute_pvalues_for_processes(self,U_matrix,chane_prob, num_bootstrapped_stats=100):
        N = U_matrix.shape[0]
        bootsraped_stats = np.zeros(num_bootstrapped_stats)

        # orsetinW = simulate(N,num_bootstrapped_stats,corr)

        for proc in range(num_bootstrapped_stats):
            # W = np.sign(orsetinW[:,proc])
            W = simulatepm(N,chane_prob)
            WW = np.outer(W, W)
            st = np.mean(U_matrix * WW)
            bootsraped_stats[proc] = N * st

        stat = N*np.mean(U_matrix)

        return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats


class QuadraticMultiple:
    def __init__(self,tester):
        self.tester = tester

    def is_from_null(self,alpha,samples,chane_prob):
        dims = samples.shape[1]
        boots = 10*int(dims/alpha)
        pvals = np.zeros(dims)
        for dim in range(dims):
            U,_ = self.tester.get_statistic_multiple_dim(samples,dim)
            p = self.tester.compute_pvalues_for_processes(U,chane_prob,boots)
            pvals[dim] = p

        print(pvals)
        alt_is_true, pvals_corrected,_,_ =  multipletests(pvals,alpha,method='holm')



        return any(alt_is_true),pvals_corrected



class QuadraticMultiple2:
    def __init__(self,tester):
        self.tester = tester

    def is_from_null(self,alpha,samples,chane_prob):
        dims = samples.shape[1]
        boots = 10*int(dims/alpha)
        pvals = np.zeros(dims)
        num_samples = samples.shape[0]
        U = np.zeros((num_samples, num_samples))
        for dim in range(dims):
            U2,_ = self.tester.get_statistic_multiple_dim(samples,dim)
            U += U2

        p = self.tester.compute_pvalues_for_processes(U,chane_prob,boots)
        return p



if __name__ == "__main__":
    sigma = np.array([[1,0.2,0.1],[0.2,1,0.4],[0.1, 0.4,1]])

    def grad_log_correleted(x):
        sigmaInv = np.linalg.inv(sigma)
        return - np.dot(sigmaInv.T + sigmaInv, x)/2.0

    me = GaussianQuadraticTest(grad_log_correleted)
    qm = QuadraticMultiple(me)
    X =  np.random.multivariate_normal([0,0,0], sigma, 200)


    reject,p_val = qm.is_from_null(0.05, X, 0.1)
    print(reject,p_val)



    qm = QuadraticMultiple2(me)
    X =  np.random.multivariate_normal([0,0,0], sigma, 200)


    p_val = qm.is_from_null(0.05, X, 0.1)
    print(p_val)


from scipy.spatial.distance import squareform, pdist

import numpy as np
from statsmodels.stats.multitest import multipletests
from stat_test.ar import simulate, simulatepm


__author__ = 'kcx, heiko'


class GaussianQuadraticTest:
    def __init__(self, grad_log_prob, scaling=2.0, grad_log_prob_multiple=None):
        self.scaling = scaling
        self.grad = grad_log_prob

        # construct (slow) multiple gradient handle if efficient one is not given
        if grad_log_prob_multiple is None:
            def grad_multiple(X):
                # simply loop over grad calls. Slow
                return np.array([self.grad(x) for x in X])

            self.grad_multiple = grad_multiple
        else:
            self.grad_multiple = grad_log_prob_multiple

    def k(self, x, y):
        return np.exp(-np.dot(x - y,x - y) / self.scaling)

    def k_multiple(self, X):
        """
        Efficient computation of kernel matrix without loops

        Effectively does the same as calling self.k on all pairs of the input
        """
        assert(X.ndim == 1)

        sq_dists = squareform(pdist(X.reshape(len(X), 1), 'sqeuclidean'))

        K = np.exp(-(sq_dists) / self.scaling)
        return K

    def k_multiple_dim(self, X):

        # check for stupid mistake
        assert X.shape[0] > X.shape[1]

        sq_dists = squareform(pdist(X, 'sqeuclidean'))

        K = np.exp(-(sq_dists) / self.scaling)
        return K


    def g1k(self, x, y):
        return -2.0 / self.scaling * self.k(x, y) * (x - y)

    def g1k_multiple(self, X):
        """
        Efficient gradient computation of Gaussian kernel with multiple inputs

        Effectively does the same as calling self.g1k on all pairs of the input
        """
        assert X.ndim == 1

        differences = X.reshape(len(X), 1) - X.reshape(1, len(X))
        sq_differences = differences ** 2
        K = np.exp(-sq_differences / self.scaling)

        return -2.0 / self.scaling * K * differences


    def g1k_multiple_dim(self, X,K,dim):

        X_dim = X[:,dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1,len(X_dim))

        return -2.0 / self.scaling * K * differences




    def g2k(self, x, y):
        return -self.g1k(x, y)

    def g2k_multiple(self, X):
        """
        Efficient 2nd gradient computation of Gaussian kernel with multiple inputs

        Effectively does the same as calling self.g2k on all pairs of the input
        """
        return -self.g1k_multiple(X)

    def g2k_multiple_dim(self, X,K,dim):
        return -self.g1k_multiple_dim(X,K,dim)

    def gk(self, x, y):
        return 2.0 * self.k(x, y) * (self.scaling - 2 * (x - y) ** 2) / self.scaling ** 2

    def gk_multiple(self, X):
        """
        Efficient gradient computation of Gaussian kernel with multiple inputs

        Effectively does the same as calling self.gk on all pairs of the input
        """
        assert X.ndim == 1

        differences = X.reshape(len(X), 1) - X.reshape(1, len(X))
        sq_differences = differences ** 2
        K = np.exp(-sq_differences / self.scaling)

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2

    def gk_multiple_dim(self, X,K,dim):
        X_dim = X[:,dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1,len(X_dim))

        sq_differences = differences ** 2

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2


    def get_statisitc(self, N, samples):
        U_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                x1 = samples[i]
                x2 = samples[j]
                a = self.grad(x1) * self.grad(x2) * self.k(x1, x2)
                b = self.grad(x2) * self.g1k(x1, x2)
                c = self.grad(x1) * self.g2k(x1, x2)
                d = self.gk(x1, x2)
                U_matrix[i, j] = a + b + c + d
        stat = N * np.mean(U_matrix)
        return U_matrix, stat


    def get_statisitc_two_dim(self, N, samples,dim):
        U_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                x1 = samples[i]
                x2 = samples[j]
                a = self.grad(x1)[dim] * self.grad(x2)[dim] * self.k(x1, x2)
                b = self.grad(x2)[dim] * self.g1k(x1, x2)[dim]
                c = self.grad(x1)[dim] * self.g2k(x1, x2)[dim]
                d = self.gk(x1, x2)[dim]
                U_matrix[i, j] = a + b + c + d
        stat = N * np.mean(U_matrix)
        return U_matrix, stat



    def get_statistic_multiple_dim(self, samples,dim):

        log_pdf_gradients = self.grad_multiple(samples)
        log_pdf_gradients = log_pdf_gradients[:,dim]
        K = self.k_multiple_dim(samples)
        G1K = self.g1k_multiple_dim(samples,K,dim)
        G2K = self.g2k_multiple_dim(samples,K,dim)
        GK = self.gk_multiple_dim(samples,K,dim)

        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U)
        return U, stat


    def get_statistic_multiple(self, samples):
        """
        Efficient statistic computation with multiple inputs

        Effectively does the same as calling self.get_statisitc.
        """
        log_pdf_gradients = self.grad_multiple(samples)
        K = self.k_multiple(samples)
        G1K = self.g1k_multiple(samples)
        G2K = self.g2k_multiple(samples)
        GK = self.gk_multiple(samples)

        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U)
        return U, stat

    def get_statistic_multiple_custom_gradient(self, samples, log_pdf_gradients):
        """
        Implements the statistic for multiple samples, each from a different
        density whose gradient at the sample is passed

        """
        K = self.k_multiple(samples)
        G1K = self.g1k_multiple(samples)
        G2K = self.g2k_multiple(samples)
        GK = self.gk_multiple(samples)

        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(len(log_pdf_gradients), 1) * log_pdf_gradients.reshape(1, len(log_pdf_gradients))
        A = pairwise_log_gradients * K
        B = G1K * log_pdf_gradients
        C = (G2K.T * log_pdf_gradients).T
        D = GK
        U = A + B + C + D
        stat = len(samples) * np.mean(U)
        return U, stat

    def compute_pvalue(self, U_matrix, num_bootstrapped_stats=100):
        N = U_matrix.shape[0]
        bootsraped_stats = np.zeros(num_bootstrapped_stats)

        for proc in range(num_bootstrapped_stats):
            W = np.sign(np.random.randn(N))
            WW = np.outer(W, W)
            st = np.mean(U_matrix * WW)
            bootsraped_stats[proc] = N * st

        stat = N*np.mean(U_matrix)

        return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats

    def compute_pvalues_for_processes(self,U_matrix,chane_prob, num_bootstrapped_stats=100):
        N = U_matrix.shape[0]
        bootsraped_stats = np.zeros(num_bootstrapped_stats)

        # orsetinW = simulate(N,num_bootstrapped_stats,corr)

        for proc in range(num_bootstrapped_stats):
            # W = np.sign(orsetinW[:,proc])
            W = simulatepm(N,chane_prob)
            WW = np.outer(W, W)
            st = np.mean(U_matrix * WW)
            bootsraped_stats[proc] = N * st

        stat = N*np.mean(U_matrix)

        return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats


class QuadraticMultiple:
    def __init__(self,tester):
        self.tester = tester

    def is_from_null(self,alpha,samples,chane_prob):
        dims = samples.shape[1]
        boots = 10*int(dims/alpha)
        pvals = np.zeros(dims)
        for dim in range(dims):
            U,_ = self.tester.get_statistic_multiple_dim(samples,dim)
            p = self.tester.compute_pvalues_for_processes(U,chane_prob,boots)
            pvals[dim] = p

        print(pvals)
        alt_is_true, pvals_corrected,_,_ =  multipletests(pvals,alpha,method='holm')



        return any(alt_is_true),pvals_corrected



class QuadraticMultiple2:
    def __init__(self,tester):
        self.tester = tester

    def is_from_null(self,alpha,samples,chane_prob):
        dims = samples.shape[1]
        boots = 10*int(dims/alpha)
        pvals = np.zeros(dims)
        num_samples = samples.shape[0]
        U = np.zeros((num_samples, num_samples))
        for dim in range(dims):
            U2,_ = self.tester.get_statistic_multiple_dim(samples,dim)
            U += U2

        p = self.tester.compute_pvalues_for_processes(U,chane_prob,boots)
        return p



if __name__ == "__main__":
    sigma = np.array([[1,0.2,0.1],[0.2,1,0.4],[0.1, 0.4,1]])

    def grad_log_correleted(x):
        sigmaInv = np.linalg.inv(sigma)
        return - np.dot(sigmaInv.T + sigmaInv, x)/2.0

    me = GaussianQuadraticTest(grad_log_correleted)
    qm = QuadraticMultiple(me)
    X =  np.random.multivariate_normal([0,0,0], sigma, 200)


    reject,p_val = qm.is_from_null(0.05, X, 0.1)
    print(reject,p_val)



    qm = QuadraticMultiple2(me)
    X =  np.random.multivariate_normal([0,0,0], sigma, 200)


    p_val = qm.is_from_null(0.05, X, 0.1)
    print(p_val)


