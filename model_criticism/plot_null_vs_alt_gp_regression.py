from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from scipy.io.matlab.mio import loadmat

import numpy as np
from stat_test.quadratic_time import GaussianQuadraticTest

import seaborn as sns
from tools.latex_plot_init import plt
sns.set_style("whitegrid")

def prepare_dataset(X, y):
    N = len(X)
    train_test_ind = int(0.9 * N)
    
    inds = np.random.permutation(N)
    X = X[inds]
    y = y[inds]
    
    # spit into train and test
    X_test = X[train_test_ind:]
    y_test = y[train_test_ind:]
    X_train = X[:train_test_ind]
    y_train = y[:train_test_ind]
    N = len(X_train)
    N_test = len(X_test)
    
    # sort for easy plotting
    temp = X_test[:, 0].argsort()
    y_test = y_test[temp]
    X_test = X_test[temp]
    temp = X[:, 0].argsort()
    y = y[temp]
    X = X[temp]
    
    # normalise by training data statistics
    X_mean = np.mean(X_train)
    X_std = np.std(X_train)
    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train -= y_mean
    y_train /= y_std
    y_test -= y_mean
    y_test /= y_std
    
    return X_train, y_train, X_test, y_test, N, N_test

def bootstrap_null(U_matrix, num_bootstrap=1000):
    bootstrapped_stats = np.empty(num_bootstrap)
    N = U_matrix.shape[0]
    
    for i in range(num_bootstrap):
        W = np.sign(np.random.randn(N))
        WW = np.outer(W, W)
        st = np.mean(U_matrix * WW)
        bootstrapped_stats[i] = N * st
    
    return bootstrapped_stats

def compute_gp_regression_gradients(y_test, pred_mean, pred_std):
    return -(y_test - pred_mean) / pred_std ** 2

def sample_null_simulated_gp(pred_mean, pred_std, num_samples=1000):
    samples = np.empty(num_samples)
    N = len(pred_mean)
    for i in range(num_samples):
        # simulate from predictive distribution and evaluate gradients at those points
        fake_y_test = np.random.randn(N) * pred_std + pred_mean
        fake_gradients = compute_gp_regression_gradients(fake_y_test, pred_mean, pred_std)
        
        # compute test statistic under this alt
        _, samples[i] = s.get_statistic_multiple_custom_gradient(fake_y_test[:, 0], fake_gradients[:, 0])
    
    return samples

if __name__ == '__main__':
    data = loadmat("../data/02-solar.mat")
    X = data['X']
    y = data['y']
    
    X_train, y_train, X_test, y_test, N, N_test = prepare_dataset(X, y)
    
    print "num_train:", len(X_train)
    print "num_test:", len(X_test)
    
    kernel = RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPRegression(X_train, y_train, kernel)
    m.optimize()
    
    res = 100
    pred_mean, pred_std = m.predict(X_test)
    plt.plot(X_test, pred_mean, 'b-')
    plt.plot(X_test, pred_mean + 2 * pred_std, 'b--')
    plt.plot(X_test, pred_mean - 2 * pred_std, 'b--')
    plt.plot(X_train, y_train, 'b.', markersize=3)
    plt.plot(X_test, y_test, 'r.', markersize=5)
    plt.grid(True)
    plt.xlabel(r"$X$")
    plt.ylabel(r"$y$")
    plt.savefig("gp_regression_data_fit.eps", bbox_inches='tight')
    plt.show()
    
    s = GaussianQuadraticTest(None)
    gradients = compute_gp_regression_gradients(y_test, pred_mean, pred_std)
    U_matrix, stat = s.get_statistic_multiple_custom_gradient(y_test[:, 0], gradients[:, 0])
    
    num_test_samples = 10000
    null_samples = bootstrap_null(U_matrix, num_bootstrap=num_test_samples)
    
    sns.distplot(null_samples, kde=False, norm_hist=True)
    plt.plot([stat, stat], [0, .012], 'black')
    plt.legend([r"$V_n$ test", r"Bootstrapped $B_n$"])
    plt.xlabel(r"$V_n$")
    plt.ylabel(r"Frequency")
    plt.savefig("gp_regression_bootstrap_hist.eps", bbox_inches='tight')
    
    plt.show()
