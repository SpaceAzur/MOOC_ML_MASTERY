# ENCORE DU TEST POUR SAVOIR QUEL ALGO EST LE MIEUX A NOTRE PROBLEME 
# MAIS CETTE FOIS, AVEC UN FOCUS SUR DES ALGORITHMES DE REGRESSION

# ==> Comment vérifier des algorithmes d'apprentissage automatique sur un problème de régression.
# A. Comment vérifier quatre algorithmes de régression linéaire
# B. Comment vérifier trois algorithmes de régression non-linéaire

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A SAVOIR : BostonHouse est chargeable directement depuis la lib sklearn :)
from sklearn.datasets import load_boston
from pandas import *
boston = load_boston()
X = boston.data 
header = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LINEAIRE
# A) 1. REGRESSION LINEAIRE
# DEF : une regression lineaire cherche à établir une relation linéaire entre une variable, dite expliquée 
#           et une ou plusieurs autres variables dite "explicatives"
# --> assume que les données d'entrées ont une distribution gaussienne
# --> assume qu'aucune variables n'ont de gros coef de corrélation entre elles

from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

filename = 'Boston.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(filename, header=1, index_col=False, names=header)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("\nA) 1. Regression Linéaire :",results.mean())

#---------------------------------------------------------------------------------------
# A) 2. RIDGE REGRESSION (dit REGULARISATION ou REGRESSION D'ARÊTES ou MATRICE DE TIKHONOV)
#   Extension de la régression linéaire où la fonction de perte est modifiée pour minimiser la complexité 
#   du modèle mesurée en tant que somme de la valeur au carré des valeurs de coefficient
#       --> technique WTF !!

from sklearn.linear_model import Ridge

array2 = dataframe.values
X2 = array2[:,0:13]
Y2 = array2[:,13]
kfold2 = KFold(n_splits=10, random_state=7)
model2 = Ridge()
scoring2 = 'neg_mean_squared_error'
results2 = cross_val_score(model2, X2, Y2, cv=kfold2, scoring=scoring2)
print("A) 2. Régression d'arêtes :",results2.mean())

#----------------------------------------------------------------------------------------
# A) 3. REGRESSION DE LASSO
#       --> technique WTF (minimisation de la complexité par les moindres carré)

from sklearn.linear_model import Lasso

array3 = dataframe.values
X3 = array3[:,0:13]
Y3 = array3[:,13]
kfold3 = KFold(n_splits=10, random_state=7)
model3 = Lasso()
scoring3 = 'neg_mean_squared_error'
results3 = cross_val_score(model3, X3, Y3, cv=kfold3, scoring=scoring3)
print("A) 3. Regression de Lasso :",results3.mean())

#---------------------------------------------------------------------------------------
# A) 4. REGRESSION ELASTICNET (combine RIDGE et LASSO)
#       --> super WTF !!!

from sklearn.linear_model import ElasticNet

array4 = dataframe.values
X4 = array4[:,0:13]
Y4 = array4[:,13]
kfold4 = KFold(n_splits=10, random_state=7)
model4 = ElasticNet()
scoring4 = 'neg_mean_squared_error'
results4 = cross_val_score(model4, X4, Y4, cv=kfold4, scoring=scoring4)
print("A) 4. Regression ElasticNet :",results4.mean())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NON-LINEAIRE
# B) 1. K-PLUS PROCHE VOISINS

from sklearn.neighbors import KNeighborsRegressor

array5 = dataframe.values
X5 = array5[:,0:13]
Y5 = array5[:,13]
kfold5 = KFold(n_splits=10, random_state=7)
model5 = KNeighborsRegressor()
scoring5 = 'neg_mean_squared_error'
results5 = cross_val_score(model5, X5, Y5, cv=kfold5, scoring=scoring5)
print("\nB) 1. k-plus proche voisins :",results5.mean())

#--------------------------------------------------------------------------------------
# B) 2. CLASSIFICATION ET ARBRE DE REGRESSION

from sklearn.tree import DecisionTreeRegressor

array6 = dataframe.values
X6 = array6[:,0:13]
Y6 = array6[:,13]
kfold6 = KFold(n_splits=10, random_state=7)
model6 = DecisionTreeRegressor()
scoring6 = 'neg_mean_squared_error'
results6 = cross_val_score(model6, X6, Y6, cv=kfold6, scoring=scoring6)
print("B) 2. arbre de decision :",results6.mean())