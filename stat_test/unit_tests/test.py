from pandas import DataFrame
import seaborn
from stat_test.quadratic_time import GaussianQuadraticTest

__author__ = 'kcx'
import numpy as  np


def grad_log_normal(x):
    return  -x


np.random.seed(42)
me = GaussianQuadraticTest(grad_log_normal)

res = np.empty((0,2))


for i in range(50):
    data = np.random.randn(75)

    _,s1 = me.get_statisitc(len(data),data)
    res = np.vstack((res,np.array([75, s1])))



for i in range(50):
    data = np.random.randn(100)
    _,s1 = me.get_statisitc(len(data),data)
    res = np.vstack((res,np.array([100, s1])))


for i in range(50):
    data = np.random.randn(150)
    _,s1 = me.get_statisitc(len(data),data)
    res = np.vstack((res,np.array([150, s1])))



df = DataFrame(res)
pr =seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()
