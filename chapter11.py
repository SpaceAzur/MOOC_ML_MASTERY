# ETUDE DES ALGORITHMES DE MACHINE LEARNING EUX-MÊMES

# ==> Vu qu'on ne sait pas à l'avance qul algo est le mieux taillé pour notre problème
# Nous voulons decouvrir quel algorithme est le plus performant sur le probleme de machine learnin étudié

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

array3 = dataframe.values
X3 = array3[:,0:8]
Y3 = array3[:,8]
kfold3 = KFold(n_splits=10, random_state=7)
model3 = KNeighborsClassifier()
results3 = cross_val_score(model3, X3, Y3, cv=kfold3)
print("3. proches voisins :", results3.mean())

#--------------------------------------------------------------------------
# 4. CLASSIFICATION NAIVE BAYESIENNE (non-linéaire)
#  --> théorème P(A|B) = P(B|A)*P(A) / P(B)
#  -> L'inférence bayésienne est une méthode par laquelle on calcule les probabilités de 
#       diverses causes hypothétiques à partir de l'observation des événements connus  
#  -> Naive Bayes calcule la probabilité de chaque classe et la probabilité conditionnelle 
#       de chaque classe en fonction de chaque valeur d'entrée.

from sklearn.naive_bayes import GaussianNB

array4 = dataframe.values
X4 = array4[:,0:8]
Y4 = array4[:,8]
kfold4 = KFold(n_splits=10, random_state=7)
model4 = GaussianNB()
results4 = cross_val_score(model4, X4, Y4, cv=kfold4)
print("4. Naïve Bayes :", results4.mean())

#-------------------------------------------------------------------------
# 5. CLASSIFICATION AND REGRESSION TREES (non-linéaire)
# --> arbre de décision (apprentissage automatique supervisé)
#   -> feuille = valeur cible de variable
#   -> embranchements = combinaison de variable d'entrée

from sklearn.tree import DecisionTreeClassifier

array5 = dataframe.values
X5 = array5[:,0:8]
Y5 = array5[:,8]
kfold5 = KFold(n_splits=10, random_state=7)
model5 = DecisionTreeClassifier()
results5 = cross_val_score(model5, X5, Y5, cv=kfold5)
print("5. Arbre de décision :", results5.mean())

#-----------------------------------------------------------------------
# 6. MACHINE A VECTEUR DE SUPPORT (non-linéaire) ou SVM
# --> identifie une ligne qui sépare le mieux deux classes
#       Les instanecs les plus proches de cette ligne sont le "vecteurs de support"

from sklearn.svm import SVC

array6 = dataframe.values
X6 = array6[:,0:8]
Y6 = array6[:,8]
kfold6 = KFold(n_splits=10, random_state=7)
model6 = SVC(gamma='auto')
results6 = cross_val_score(model6, X6, Y6, cv=kfold6)
print("6. Vecteur de support :", results6.mean())


#=====================================================================================
# CONCLUSION 
# Ici l'algorithme 2. Analyse discriminate linéaire est le plus performant sur le jeu de 
# données 'pima-indians-diabetes'
#=====================================================================================
