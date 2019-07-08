# COMPARER LES ALGORITHMES DE MACHINE LEARNING ENTRE EUX 

# ==> CREER UN JEU DE TEST POUR COMPARER LES ALGO ENTRE EUX ==> COMPARAISON HOMOGENE 
#           On teste les algo avec la même methode, ici Kflod (où k et itération identiques entre les algo)

from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# chargement des données
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# preeparation des modele à comparer
models = []
models.append(('Regression logistique', LogisticRegression(solver='liblinear')))
models.append(('Analyse discriminante linéaire', LinearDiscriminantAnalysis()))
models.append(('K-Plus proche voisin', KNeighborsClassifier()))
models.append(('Arbre de decision', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('vecteur de support', SVC(gamma='auto')))

# Evaluer chaque model
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# visualisation des resultats
fig = pyplot.figure()
fig.suptitle("Comparaison d'algorithme")
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
