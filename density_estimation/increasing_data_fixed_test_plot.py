import os

import numpy as np
import pandas as pd
import seaborn as sns
from tools.latex_plot_init import plt


fname = "increasing_data_fixed_test.txt"

fields = ['p_value']
field_plot_names = {
                    'p_value': 'p-value',
                    'N': r'$N$'
                    }
def kwargs_gen(**kwargs):
    return kwargs

conditions = kwargs_gen(
                          D=1,
                          N_test=500,
                          num_bootstrap=200,
                          sigma=1,
                          lmbda=0.01,
                        )

# x-axis of plot
x_field = 'N'
x_field_values = [50, 100, 500, 1000, 2000, 5000]

df = pd.read_csv(fname, index_col=0)

for field in fields:
    plt.figure()
    
    # filter out desired entries
    mask = (df[field] == df[field])
    for k,v in conditions.items():
        mask &= (df[k] == v)
    current = df.loc[mask]
    
    # only use desired values of x_fields
    current = current.loc[[True if x in x_field_values else False for x in current[x_field]]]
    
    # use ints on x-axis
    current[x_field] = current[x_field].astype(int)
    
    sns.set_style("whitegrid")
    sns.boxplot(x=x_field, y=field, data=current.sort(x_field))

    plt.xlabel(field_plot_names[x_field])
    plt.ylabel(field_plot_names[field])

    fname_base = os.path.splitext(fname)[0]
    plt.savefig(fname_base + ".png", bbox_inches='tight')
    plt.savefig(fname_base + ".eps", bbox_inches='tight')
    
    # print info on number of trials
    print(field)
    print("Average number of trials: %d" % int(np.round(current.groupby(x_field).apply(len).mean())))
    print(current.groupby(x_field).apply(len))
    
plt.show()
