import numpy as np
import seaborn
from pandas import DataFrame

from tools.latex_plot_init import plt

results = np.load('results_good.npy')
df = DataFrame(results)

plt.figure()

seaborn.set_style("whitegrid")
seaborn.boxplot(x=0, y=1, data=df,palette="BuGn_d")

plt.tight_layout()
plt.ylabel('p values')
plt.ylim([0,1])
plt.xlabel('degrees of freedom')
plt.savefig('../write_up/img/sgld_student.pdf')

results = np.load('results_bad.npy')
df = DataFrame(results)

plt.figure()

seaborn.set_style("whitegrid")
seaborn.boxplot(x=0, y=1, data=df,palette="BuGn_d")

plt.tight_layout()
plt.ylabel('p values')
plt.ylim([0,1])
plt.xlabel('degrees of freedom')
plt.savefig('../write_up/img/sgld_student_bad.pdf')


results = np.load('results_thinning.npy')
df = DataFrame(results)

plt.figure()

seaborn.set_style("whitegrid")
seaborn.boxplot(x=0, y=1, data=df,palette="BuGn_d")

plt.tight_layout()
plt.ylabel('p values')
plt.ylim([0,1])
plt.xlabel('degrees of freedom')
plt.savefig('../write_up/img/sgld_student_opt.pdf')