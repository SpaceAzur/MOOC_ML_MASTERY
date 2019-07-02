'''
Resoudre les problèmes courants de fuite de données lorsqu'on compare des algorithmes 
                                                                  d'apprentissage automatique.
OBJECTIFS
==> automatiser les workflow de Machine Learning : UTILISATION DES PIPELINES de sklearn
PIPELINE permet de chaîner plusieurs estimateur en un. 
Utile car existe souvent une sequence fixe d'étapes dans le traitement des data

ABSTRACT (important)
La préparation des données est un moyen facile de perdre des connaissances sur l’ensemble des données 
qui serviront à ou aux algorithme(s). Par exemple, préparer les données à l'aide de la standardisation ou de la 
normalisation sur l'intégralité du jeu de données d'apprentissage avant l'apprentissage ne constitue pas 
un test valide, car celui-ci aurait été influencé par la mise à l'échelle.

Les pipelines vous aident à prévenir les fuites de données dans votre faisceau de tests en garantissant 
que la préparation des données, telle que la normalisation, est limitée à chaque étape de votre procédure 
de validation croisée.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
2 etapes pour définir le pipelines
--> 1. Standardiser les données
--> 2. Apprentissage et Analyse Discriminante Linéaire
On utilisera 10-folds cross validation
'''

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# chargement des données
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# creation du pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

# evaluer le pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Pipeline :",results.mean())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UTILISER FEATUREUNION
'''
Le pipeline fournit un outil pratique appelé FeatureUnion qui permet de combiner les résultats de plusieurs 
procédures de sélection et d'extraction de caractéristiques dans un jeu de données plus grand sur lequel un 
modèle peut être formé
'''
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# create feature union
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators2 = []
estimators2.append(('feature_union', feature_union))
estimators2.append(('logistic', LogisticRegression(solver='liblinear')))
model2 = Pipeline(estimators2)
# evaluate pipeline
kfold2 = KFold(n_splits=10, random_state=7)
results2 = cross_val_score(model2, X, Y, cv=kfold2)
print("Pipeline + FeatureUnion :", results2.mean())

import nltk
nltk.download()


