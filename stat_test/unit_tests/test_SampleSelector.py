from unittest import TestCase
from sampplers.MetropolisHastings import mh_generator
from stat_test.linear_time import SampleSelector, GaussianSteinTest
import numpy as np

__author__ = 'kcx'


class TestSelector(TestCase):




    def test_on_one_dim_gaussian(self):
        np.random.seed(42)

        def log_normal(x):
                return  -np.dot(x,x)/2

        generator = mh_generator(log_density=log_normal)

        def gradient_of_log_of_normal(x):
            return  -x


        tester = GaussianSteinTest(gradient_of_log_of_normal,10)


        selector = SampleSelector(generator, sample_size=2000,thinning=15,tester=tester)

        data,converged = selector.points_from_stationary()

        tester = GaussianSteinTest(gradient_of_log_of_normal,10)
        assert tester.compute_pvalue(data)>0.05
        assert converged


    def test_with_fake_log_prob(self):
        np.random.seed(42)


        def grad_log_prob(x):
            return -(x/2.0 + np.sin(x))*(1.0/2.0 + np.cos(x))

        def fake_log_prob(x):
            return -(x/5.0 + np.sin(x) )**2.0/2.0

        generator = mh_generator(log_density=fake_log_prob,x_start=1.0)
        tester = GaussianSteinTest(grad_log_prob,41)

        selector = SampleSelector(generator, sample_size=1000,thinning=20,tester=tester, max_iterations=5)

        data,converged = selector.points_from_stationary()

        assert converged is False

    def test_with_ugly(self):
        np.random.seed(42)


        def grad_log_prob(x):
            return -(x/5.0 + np.sin(x))*(1.0/5.0 + np.cos(x))

        def log_prob(x):
            return -(x/5.0 + np.sin(x) )**2.0/2.0

        generator = mh_generator(log_density=log_prob,x_start=1.0)
        tester = GaussianSteinTest(grad_log_prob,41)

        selector = SampleSelector(generator, sample_size=1000,thinning=20,tester=tester, max_iterations=5)

        data,converged = selector.points_from_stationary()

        tester = GaussianSteinTest(grad_log_prob,41)
        assert tester.compute_pvalue(data)>0.05

        assert converged is True
