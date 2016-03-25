import warnings
import numpy as np
from numpy.linalg import linalg, LinAlgError
from scipy.stats import chi2

POWER = 2.0

__author__ = 'kcx'


def mahalanobis_distance(difference, num_random_features):
    num_samples, _ = np.shape(difference)
    sigma = np.cov(np.transpose(difference))

    mu = np.mean(difference, 0)

    if num_random_features == 1:
        stat = float(num_samples * mu ** 2) / float(sigma)
    else:
        try:
            linalg.inv(sigma)
        except LinAlgError:
            print('covariance matrix is singular. Pvalue returned is 1.1')
            warnings.warn('covariance matrix is singular. Pvalue returned is 1.1')
            return 0
        stat = num_samples * mu.dot(linalg.solve(sigma, np.transpose(mu)))

    return chi2.sf(stat, num_random_features)

class GaussianSteinTest:
    def __init__(self, grad_log_prob, num_random_freq, scaling=(1.0, 10.0)):
        self.number_of_random_frequencies = num_random_freq
        self.scaling = scaling

        def stein_stat(random_frequency, samples):
            random_scale = np.random.uniform(self.scaling[0], self.scaling[1])
            a = grad_log_prob(samples)
            b = self._gaussian_test_function(samples, random_frequency, random_scale)
            c = self._test_function_grad(samples, random_frequency, random_scale)
            return a * b + c

        self.stein_stat = stein_stat


    def _make_two_dimensional(self, z):
        if len(z.shape) == 1:
            z = z[:, np.newaxis]
        return z

    def _get_mean_embedding(self, x, random_frequency, scaling=2.0):
        z = x - random_frequency
        z = linalg.norm(z, axis=1) ** POWER
        z = np.exp(-z / scaling)
        return z

    def _gaussian_test_function(self, x, random_frequency, scaling=2.0):
        x = self._make_two_dimensional(x)
        mean_embedding = self._get_mean_embedding(x, random_frequency, scaling)
        return np.tile(mean_embedding, (self.shape, 1)).T


    def _test_function_grad(self, x, omega, scaling=2.0):
        arg = (x - omega) * POWER / scaling
        test_function_val = self._gaussian_test_function(x, omega, scaling)
        return -arg * test_function_val


    def compute_pvalue(self, samples):

        samples = self._make_two_dimensional(samples)

        self.shape = samples.shape[1]

        stein_statistics = []


        for f in range(self.number_of_random_frequencies):
            # This is a little bit of a bug , but th holds even for this choice
            random_frequency = np.random.randn()
            matrix_of_stats = self.stein_stat(random_frequency=random_frequency, samples=samples)
            stein_statistics.append(matrix_of_stats)

        normal_under_null = np.hstack(stein_statistics)
        normal_under_null = self._make_two_dimensional(normal_under_null)

        return mahalanobis_distance(normal_under_null, normal_under_null.shape[1])


class SampleSelector:
    def __init__(self, data_generator, sample_size, thinning, tester, alpha=0.05, max_iterations=10):
        self.data_generator = data_generator
        self.thinning = thinning
        self.tester = tester
        self.n = sample_size
        self.alpha = alpha
        self.max_ite = max_iterations

    def points_from_stationary(self):

        zeta2 = 1.645
        indicator = 0.0
        stop = False
        while not stop:
            indicator +=  1
            test_level = 1 / (indicator ** 2) * (1 / zeta2) * self.alpha
            data = self.data_generator.get(self.n, self.thinning)

            out_of_iterations = indicator > self.max_ite
            accept_null = self.tester.compute_pvalue(data) > test_level
            stop = out_of_iterations or accept_null

        if out_of_iterations:
            msg = 'The sample is NOT from stationary distribution. Exceeded number of iterations until convergence.'
            warnings.warn(msg)

        converged = not out_of_iterations
        return data, converged


