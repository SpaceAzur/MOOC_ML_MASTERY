# PRACTICE WITH AN END-TO-END MACHINE LEARNING PROJECT

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ==> 6 TÂCHES FORME UN PROJET

#       1. Definir le problème
#       2. Résumé les données
#       3. Préparer les données
#       4. Evaluer les algorithmes
#       5. Améliorer les résultats
#       6. Présenter les resultats / Finaliser le modèle

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Plus en détail :

# 1. Prepare Problem
#       a) Load libraries
#       b) Load dataset

# 2. Summarize Data
#       a) Descriptive statistics
#       b) Data visualizations

# 3. Prepare Data
#       a) Data Cleaning
#       b) Feature Selection
#       c) Data Transforms

# 4. Evaluate Algorithms
#       a) Split-out validation dataset
#       b) Test options and evaluation metric
#       c) Spot Check Algorithms
#       d) Compare Algorithms

# 5. Improve Accuracy
#       a) Algorithm Tuning
#       b) Ensembles

# 6. Finalize Model
#       a) Predictions on validation dataset
#       b) Create standalone model on entire training dataset
#       c) Save model for later use

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1 Prepare Problem : loading data

import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

filename = 'iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width','class']
dataset = read_csv(filename, names=names)

#shape
nb_ligne, nb_colonne = dataset.shape
print("\nnb ligne (instance) %d \nnb colonne (classe) %d" % (nb_ligne,nb_colonne))
# description
print("\nDescription :\n", dataset.describe())
# class distribution
print("Distribution par classe\n",dataset.groupby('class').size())

# UNIVARIATE PLOTS : box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# MULTIVARIATE PLOTS : scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Schematiser la correlation entre les classes
correlation =dataset.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# reprendre p.118 chapitre 19.5.1