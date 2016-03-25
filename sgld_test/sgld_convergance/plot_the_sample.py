from sgld_test.mcmc_convergance.cosnt import CHAIN_SIZE, SGLD_CHAIN_SIZE

__author__ = 'kcx'
import  numpy as np

samples = np.load('./samples.npy')


last = samples[:,SGLD_CHAIN_SIZE-1]

print(last)

import seaborn as sns
sns.set(color_codes=True)
with sns.axes_style("white"):
    pr = sns.jointplot(x=last[:,0], y=last[:,1], kind="hex", color="k");


sns.plt.show()
pr.savefig('../../write_up/img/sgld_sample.pdf')

