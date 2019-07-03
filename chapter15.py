'''
METHODE DES ENSEMBLE !! pour améliorer la performance des algo

OBJECTIFS : Combiner les modèles dans des ensembles prédictifs
==> 3 approches pour les ensembles :
A ) BAGGING | METHODE D'AGGREGATION DE MODELE
B ) BOOSTING | ALGORITHMES DE BOOSTING
C ) VOTING ENSEMBLE | ENSEMBLE DE VOTE

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
#############################
######### BAGGING ###########
############################# 

# A) 1. BAGGING | METHODE D'AGGREGATION DE MODELE
# ==> prend plusieurs echantillon des data et entraine chaque echantillon avec un modele
#       output final = moyenne des predictions de chaque modele

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print("\nA) 1. Bagging :",results.mean())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A) 2. RANDOM FOREST
#  --> idem bagging mais chaque echantillon est construit pour avoir le moins de correlation possible entre eux

from sklearn.ensemble import RandomForestClassifier

array2 = dataframe.values
X2 = array2[:,0:8]
Y2 = array2[:,8]
num_trees2 = 100
max_features2 = 3
kfold2 = KFold(n_splits=10, random_state=7)
model2 = RandomForestClassifier(n_estimators=num_trees2, max_features=max_features2)
results2 = cross_val_score(model2, X2, Y2, cv=kfold2)
print("A) 2. Random forest :",results2.mean())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A) 3. EXTRA TREES
#  --> idem random forest mais plus WTF

from sklearn.ensemble import ExtraTreesClassifier

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 100
max_features = 7
kfold = KFold(n_splits=10, random_state=7)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print("A) 3. Extra Trees :",results.mean())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#############################
######### BOOSTING ##########
############################# 

# ==> ensemble de boosting créé une sequence de modele, chaque modèle tentant de 
#                                                               corriger les erreurs du précédent

# B) 1. AdaBoost
# --> pondère chaque données du jeu selon sa difficulté à être classé, permet à l'algo d'y accorder + ou - 
#                                                                  d'importance dans la séquence.

from sklearn.ensemble import AdaBoostClassifier

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
num_trees = 30
seed=7
kfold = KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print("\nB) 1. AdaBoost :",results.mean())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# B) 2. STOCHASTIC GRADIENT BOOSTING (descente de gradient stochastique)
# --> rien compris, mais considéré comme méthode la plus efficace

from sklearn.ensemble import GradientBoostingClassifier

array3 = dataframe.values
X3 = array3[:,0:8]
Y3 = array3[:,8]
seed3 = 7
num_trees3 = 100
kfold3 = KFold(n_splits=10, random_state=seed)
model3 = GradientBoostingClassifier(n_estimators=num_trees3, random_state=seed3)
results3 = cross_val_score(model3, X3, Y3, cv=kfold3)
print("B) 2. Gradient stochastique :",results3.mean())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#############################
##### VOTING ENSEMBLE #######
############################# 

# ==> facile pour cumuler les prédiction de plusieurs algo

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)

# creer les sous model (algo different)
estimators = []
modelA = LogisticRegression(solver='liblinear')
estimators.append(('logistic', modelA))
modelB = DecisionTreeClassifier()
estimators.append(('cart', modelB))
modelC = SVC(gamma='auto')
estimators.append(('svm', modelC))

# créer le modèle d'ensemble
ensemble = VotingClassifier(estimators)
results4 = cross_val_score(ensemble, X, Y, cv=kfold)
print("\nC) Model d'ensemble :",results4.mean())


