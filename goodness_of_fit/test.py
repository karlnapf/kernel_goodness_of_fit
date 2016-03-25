from scipy.spatial.distance import squareform, pdist

import numpy as np

__author__ = 'kcx, heiko'


def simulatepm(N, p_change):
    '''

    :param N:
    :param p_change:
    :return:
    '''
    X = np.zeros(N) - 1
    change_sign = np.random.rand(N) < p_change
    for i in range(N):
        if change_sign[i]:
            X[i] = -X[i - 1]
        else:
            X[i] = X[i - 1]
    return X


class _GoodnessOfFitTest:
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

    def kernel_matrix(self, X):

        # check for stupid mistake
        assert X.shape[0] > X.shape[1]

        sq_dists = squareform(pdist(X, 'sqeuclidean'))

        K = np.exp(-sq_dists/ self.scaling)
        return K

    def gradient_k_wrt_x(self, X, K, dim):

        X_dim = X[:, dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1, len(X_dim))

        return -2.0 / self.scaling * K * differences

    def gradient_k_wrt_y(self, X, K, dim):
        return -self.gradient_k_wrt_x(X, K, dim)

    def second_derivative_k(self, X, K, dim):
        X_dim = X[:, dim]
        assert X_dim.ndim == 1

        differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1, len(X_dim))

        sq_differences = differences ** 2

        return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2

    def get_statistic_multiple_dim(self, samples, dim):
        num_samples = len(samples)

        log_pdf_gradients = self.grad_multiple(samples)
        log_pdf_gradients = log_pdf_gradients[:, dim]
        K = self.kernel_matrix(samples)
        gradient_k_x = self.gradient_k_wrt_x(samples, K, dim)
        gradient_k_y = self.gradient_k_wrt_y(samples, K, dim)
        second_derivative = self.second_derivative_k(samples, K, dim)

        # use broadcasting to mimic the element wise looped call
        pairwise_log_gradients = log_pdf_gradients.reshape(num_samples, 1) \
                                 * log_pdf_gradients.reshape(1, num_samples)
        A = pairwise_log_gradients * K

        B = gradient_k_x * log_pdf_gradients
        C = (gradient_k_y.T * log_pdf_gradients).T
        D = second_derivative

        V_statistic = A + B + C + D

        stat = num_samples * np.mean(V_statistic)
        return V_statistic, stat

    def compute_pvalues_for_processes(self, U_matrix, chane_prob, num_bootstrapped_stats=100):
        N = U_matrix.shape[0]
        bootsraped_stats = np.zeros(num_bootstrapped_stats)

        for proc in range(num_bootstrapped_stats):
            # W = np.sign(orsetinW[:,proc])
            W = simulatepm(N, chane_prob)
            WW = np.outer(W, W)
            st = np.mean(U_matrix * WW)
            bootsraped_stats[proc] = N * st

        stat = N * np.mean(U_matrix)

        return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats


class GoodnessOfFitTest:
    def __init__(self, grad_log_prob, scaling=2.0, grad_log_prob_multiple=None):
        self.tester = _GoodnessOfFitTest(grad_log_prob, scaling, grad_log_prob_multiple)

    def is_from_null(self, alpha, samples, chane_prob):
        dims = samples.shape[1]
        boots = 10 * int(dims / alpha)
        num_samples = samples.shape[0]
        U = np.zeros((num_samples, num_samples))
        for dim in range(dims):
            U2, _ = self.tester.get_statistic_multiple_dim(samples, dim)
            U += U2

        p = self.tester.compute_pvalues_for_processes(U, chane_prob, boots)
        return p


if __name__ == "__main__":
    sigma = np.array([[1, 0.2, 0.1], [0.2, 1, 0.4], [0.1, 0.4, 1]])


    def grad_log_correleted(x):
        sigmaInv = np.linalg.inv(sigma)
        return - np.dot(sigmaInv.T + sigmaInv, x) / 2.0


    me = _GoodnessOfFitTest(grad_log_correleted)

    qm = GoodnessOfFitTest(me)
    X = np.random.multivariate_normal([0, 0, 0], sigma, 200)

    p_val = qm.is_from_null(0.05, X, 0.1)
    print(p_val)
