# IMPROVE PERFORMANCE WITH ALGORITHMS TUNING
# ==> Comment adapter les paramètres des modeles de Machine Learning
#     2 méthodes pour adapter les paramètres

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. GRID SEARCH PARAMETERS TUNING
#    --> Réglage des paramètres de recherche dans la grille (grille de parametre)
#        la méthode va recherche la meilleur combinaison de paramètre présent dans la grille

import numpy
from pandas import read_csv
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = RidgeClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(X, Y)
print("1. Resultat optimal :", grid.best_score_)
print(" -> valeur du paramètre pour resultat optimal :",grid.best_estimator_.alpha)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. RANDOM SEARCH PARAMETER TUNING
# --> La recherche aléatoire est une approche de réglage de paramètre qui 
#       échantillonne les paramètres d'algorithme à partir d'une distribution aléatoire 
#       (c'est-à-dire uniforme) pour un nombre fixe d'itérations

import scipy
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
param_grid = {'alpha': uniform()}
model = RidgeClassifier()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100,
cv=3, random_state=7)
rsearch.fit(X, Y)
print("\n2. Resultat optimal :",rsearch.best_score_)
print(" -> valeur du paramètre pour resultat optimal :",rsearch.best_estimator_.alpha)

