# Understand the data and visualize them
import requests, csv, numpy as np, pandas
from numpy import *
from pandas import *
from matplotlib import pyplot


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Histogramme des attributs
data.hist()
pyplot.show()

# courbe de denisté des attributs
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()

# Box and Whisker Plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

# plot correlation matrix
correlations = data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()