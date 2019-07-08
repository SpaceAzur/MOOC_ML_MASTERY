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

# séparer les data en un jeu d'entrainement, et un autre jeu de validation pour plus tard

array = dataset.values
X = array[:,0:4] #ne prend pas en compte la dernière colonne [5] 
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, 
                                test_size=validation_size, random_state=seed)

print("X_train :",len(X_train), "\nX_validation :",len(X_validation))
# harnais de test 
#   nous allons eclater nos données en 10 sous parties (10 folders) 
#   le modele va s'entrainer sur 9 d'entre elles et tester sur le dernier folder 
#   Il va itérer cette opération sur toutes les combinaisons de folder existantes

# construire les modeles 
#   nous allons faire cela sur plusieurs modeles (6) afin de choisir lequel est le plus performant/pertinent
#   avec le même traitement des données (folder) pour chaque model, afin de les rendre comparable

models =[]
models.append(('LogisticRegression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('+ProcheVoisins', KNeighborsClassifier()))
models.append(('ArbreDecision', DecisionTreeClassifier()))
models.append(('NaiveBayes', GaussianNB()))
models.append(('VecteurDeSupport', SVC(gamma='auto')))

# on évalue chaque modele
results = []
names = []
 
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
    print(msg)
        # Resultats /                      Précision | Variance
            # LogisticRegression:           0.966667  (0.040825)
            # LinearDiscriminantAnalysis:   0.975000  (0.038188)
            # +ProcheVoisins:               0.983333  (0.033333)
            # ArbreDecision:                0.975000  (0.038188)
            # NaiveBayes:                   0.975000  (0.053359)
            # VecteurDeSupport:             0.991667  (0.025000)

# Comparer les algo ==> schéma representatif

fig = pyplot.figure()
fig.suptitle("Comparaison d'algorithtmes")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# faire des prédictions sur le jeu de données de validation
svc = SVC(gamma='auto')
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print("\naccuracy :\n", accuracy_score(Y_validation, predictions))
print("\nmatrice de confusion :\n", confusion_matrix(Y_validation, predictions))
print("\nrapport de classification :\n", classification_report(Y_validation, predictions))



