import requests, pandas, io, csv, re
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# url = 'https://raw.githubusercontent.com/selva86/datasets/master/Sonar.csv'
# r = requests.get(url)
# with open('Sonar.csv', 'w') as f:         ~~TELECHARGEMENT ET SAUVEGARDE DES DONNEES~~
#     writer = csv.writer(f)                    ~~> n'est fait qu'une seule fois~~
#     for line in r.iter_lines():
#         writer.writerow(line.decode('utf-_').split(','))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NATURE DES DONNES : RETOUR DE SIGNAL RADAR (projet de données binaires)
# OBJECTIF          : PREDIRE SI UN OBJET EST METALEUX(0) OU ROCHEUX(1)
'''Chaque ligne de "Sonar.csv" est un ensemble de 60 nombres compris entre 0,0 et 1,0. Chaque nombre 
représente l'énergie dans une bande de fréquences particulière, intégrée sur une certaine période. 
L'étiquette associée à chaque enregistrement contient la lettre R si l'objet est une roche
 et M si c'est du métal'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# importation des données
# donnees = pandas.read_csv('Sonar.csv', index_col=False, header=None)
data = pandas.read_csv('Sonar3.csv', header=None)

# gestion de l'entête
    # entete = []
    # for i in donnees.iloc[0]:
    #     entete.append(i)
    # entete = [i.replace('"','') for i in entete]
entete = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
'V13', 'V14', 'V15', 'V16', 'V17', 'V18','V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 
'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40',
 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 
 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'Class']

# dimensions des données => 208 instances et 61 colonnes
print(data.shape)

# types des données
# set_option('display.max_rows', 500)
# print(data.dtypes)

set_option('display.width', 100)
print(data.head(20))

# description des donnees
set_option('precision', 3)
print(data.describe())

# # schématisation d'un attribut par nuage de points
# x = range(208)
# y = data[54]
# pyplot.scatter(x,y)
# pyplot.title('nuage de points colonne 2')
# pyplot.xlabel('x')
# pyplot.ylabel('y')
# pyplot.show()

# distribution entre instances rocheuses(1) et métalleuses(0)
print(data.groupby(60).size())

# visualisation des donnees | histogramme
data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()

# visualisation des donnees | courbe de densité
data.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=False, fontsize=1)
pyplot.show()

''' nous voyons que beaucoup d'attributs ont une distribution asymétrique
Nous pourrons tenter de transformer les données par le jeu des puissances pour corriger cela'''

# matrice de correlation
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
pyplot.show()

''' nous voyons que les attributs voisins entre eux ont tendances à correlation
au contraire de ceux qui sont loin des uns de sautres, dont leur correlation s'eloigne significativement
CELA A DU SENS considérant que l'ordre des attributs suit l'angle des signaux récupérés par le radar'''

# creation du jeu de validation et du jeu d'entrainement
array = data.values
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_size = 0.20 # jeu de test = 20% des data disponibles
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# options de test et evaluation métrique
nb_de_fois = 10
seed = 7
scoring= 'accuracy'


# algo de spot-check
modele = []
modele.append(('\nRegression Lineaire', LogisticRegression(solver='liblinear')))
modele.append(('Analyse Discriminante Linaire', LinearDiscriminantAnalysis()))
modele.append(('+Proche Voisins', KNeighborsClassifier()))
modele.append(('Arbre de decision', DecisionTreeClassifier()))
modele.append(('Naive Bayes', GaussianNB()))
modele.append(('vecteur de support SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in modele:
    kfold = KFold(n_splits=nb_de_fois, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %.3f (%.3f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Comparaison des distribution des Algorithmes de régression
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

''' +Proche Voisins est le candidat idéal
=> a la meilleure précision (~80%) et la plus faible dispersion'''

# standardisation des données ==> Pipelines
pipelines = []
pipelines.append(('\nS_Regression_Lineaire', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression(solver='liblinear'))])))
pipelines.append(('S_Analyse_discrimante_lineaire', Pipeline([('Scaler', StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))
pipelines.append(('S_+Proche_Voisins', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsClassifier())])))
pipelines.append(('S_Arbre_de_decision', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeClassifier())])))
pipelines.append(('SçNaive_Bayes', Pipeline([('Scaler', StandardScaler()),('NB',GaussianNB())])))
pipelines.append(('S_Vecteur_de_support', Pipeline([('Scaler', StandardScaler()),('SVM',SVC(gamma='auto'))])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=nb_de_fois, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Comparaison des distributions standardisées des Algorithmes de régression
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

''' nous voyons que Vecteur de support est aussi très pertinent (en plus de +ProchesVoisins)'''

# Règlages des paramètres de +Proches_Voisins
'''le nb de voisins par default est de 7, tentons avec des valeurs entre 1 et 21'''
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
voisins = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=voisins)
modele2 = KNeighborsClassifier()
kfold2 = KFold(n_splits=nb_de_fois, random_state=seed)
grid = GridSearchCV(estimator=modele2, param_grid=param_grid, scoring=scoring, cv=kfold2, iid=True)
grid_result = grid.fit(rescaledX, Y_train)

print("\nBest: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

'''Nous pouvons voir que la configuration optimale est K = 1. Ceci est intéressant car 
l'algorithme fera des prédictions en utilisant l'instance la plus similaire dans le seul 
jeu de données d'apprentissage'''

# Règlages des paramètres des Vecteurs de Support (ou dit Séparateur à vaste marges)
'''nous pouvons régler 2 paramètres du SVM => C (détente de la marge) et Kernel (noyau)
MARGE : est la distance entre la frontière de séparation et les échantillons les plus proches
        est la plus petites distances 'signée' entre la surface de décision (la droite de régression)
        et les entrées de l'ensemble d'entrainement 
            => LE SVM CHERCHE A MAXIMISER LA MARGE
            ==> nous paramétrons C = {0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0}
KERNEL: Les fonctions noyau permettent de transformer un produit scalaire dans un espace de grande 
        dimension, ce qui est coûteux{=> O(N^3)}, en une simple évaluation ponctuelle d'une fonction
        EST COUTEUX. Utiliser quand nx input ne peut pas être classifier par une discrimination
        linéaire spéarables. DOnc nous recherchons des dimensions supérieur au problème pour voir
        si, dans ce nouvel espace, il existe une separation linéaire pour le nouvel input.
        Cela reduit sensiblement le coût de calcul.
            => NOYAU permet de calculer dans l'espace de calcul initial
            ==> nous allons tester toutes les valeurs possible du paramètre "kernel"
'''

scaler2 = StandardScaler().fit(X_train)
rescaledX2 = scaler2.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid2 = dict(C=c_values, kernel=kernel_values)
modele3 = SVC(gamma='auto')
kfold3 = KFold(n_splits=nb_de_fois, random_state=seed)
grid2 = GridSearchCV(estimator=modele3, param_grid=param_grid2, scoring=scoring, cv=kfold3, iid=True)
grid_result2 = grid2.fit(rescaledX2, Y_train)


print("\nBest: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))
means = grid_result2.cv_results_['mean_test_score']
stds = grid_result2.cv_results_['std_test_score']
params = grid_result2.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# reprednre p.159 méthode des ensembles