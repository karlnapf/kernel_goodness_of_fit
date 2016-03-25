from sgld_test.mcmc_convergance.cosnt import CHAIN_SIZE
import numpy as np
__author__ = 'kcx'


samples = np.load('./samples.npy')


last = samples[:,CHAIN_SIZE-1]

print(last)

import seaborn as sns
sns.set(color_codes=True)
with sns.axes_style("white"):
    pr = sns.jointplot(x=last[:,0], y=last[:,1], kind="hex", color="k");
    # fig = pr.get_figure()
    pr.savefig('../../write_up/img/mcmc_sample.pdf')

    sns.plt.show()