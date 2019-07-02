# ETUDE DES ALGORITHMES DE MACHINE LEARNING EUX-MÊMES

#-------------------------------------------------------------
#-> est une manière pour savoir quel algorithme sera le plus adapté et performant à un problème
# -> La question ici n'est pas : Quel algorithme dois-je utilisé sur mon jeu de données ?
#       Mais plutôt : Quel algorithme dois-je vérifier sur mon jeu de données ?


# =====> 6 algorithmes de vérification pouvant être utilisé sur nos données <==========

# 1. REGRESSION LOGICTIQUE (linéaire)
#       --> trouve la valeur d'une variable Y grâce à l'analyse d'autres variables
#       --> variable à valeur binaire (0 ou 1)

from pandas import read_csv
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, Y, cv=kfold)
print("1. Regression Logistique :", results.mean())
        # valeur entre 0 et 1, proche de 1 = pertinent

#-------------------------------------------------------------------------
# 2. ANALYSE DISCRIMINANTE LINEAIRE (linéaire)
#       --> expliqueret prédire l'appartenancec d'un individu à une classe (groupe) prédéfinie
#               à partir de ses caractéristiques mesurées à l'aide de variables prédictives

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

array2 = dataframe.values
X2 = array2[:,0:8]
Y2 = array2[:,8]
kfold2 = KFold(n_splits=10, random_state=7)
model2 = LinearDiscriminantAnalysis()
results2 = cross_val_score(model2, X2, Y2, cv=kfold2)
print("2. Analyse discriminante linéaire :", results2.mean())

#--------------------------------------------------------------------------
# 3. METHODE DES K-PLUS PROCHES VOISINS (non-linéaire)
#   -> trouve les k instances les plus proches entre elles (valeurs) 
#                                       dans les données et compare leurs voisins

from sklearn.neighbors import KNeighborsClassifier

