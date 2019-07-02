# INDICATEURS DE PERFORMANCE -> comment les choisir

# Existe 5 indicateurs/estimateurs

#----------------------------------------------------------------
# 1. Classification Accuracy
#   --> nb de prédictions correcte / nb total de prédiction
#   --> ATTENTION pour être pertinente, le nn d'observation de chaque classe doit être le même !!

import pandas
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("1. Accuracy : %.3f (%.3f)" % (results.mean(), results.std()))

#-------------------------------------------------------------------
# 2. LOGARITHMIC LOSS (est une mesure de performance permettant d'évaluer les prédictions 
#                                                       de probabilités d'appartenance à une classe donnée.)
#   --> sera mesuré par un indicateur de confiance entre 0 et 1
#   --> Les prédictions qui sont correctes ou incorrectes sont récompensées ou punies 
#                                           proportionnellement à la confiance de la prédiction.

kfold2 = KFold(n_splits=10, random_state=7)
model2 = LogisticRegression(solver='liblinear')
scoring = 'neg_log_loss'
results2 = cross_val_score(model2, X, Y, cv=kfold, scoring=scoring)
print("2. LogLoss : %.3f (%.3f)" % (results2.mean(), results2.std()))
                    # le resultat LogLosse est inversé (car < à 0) par la fonction cross_valçscore, 
                    # mais ne peut pas être négative, donc ici --> 0.493
                    # plus le résultat est proche de 0 et mieux c'est

#---------------------------------------------------------------------------
# 3. AREA UNDER ROC CURVE (Courbe ROC, théorie des signaux et classification binaire, 
#                                                               pour éliminer le bruit de fond)
#   --> voir wikipedia Courbe ROC

kfold3 = KFold(n_splits=10, random_state=7)
model3 = LogisticRegression(solver='liblinear')
scoring2 = 'roc_auc'
results3 = cross_val_score(model3, X, Y, cv=kfold3, scoring=scoring2)
print("3. AUC : %.3f (%.3f)" % (results3.mean(), results3.std()))

#-----------------------------------------------------------------------------
# 4. CONFUSION MATRIX (voir wikipedia, très bon article "matrice de confusion")

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model4 = LogisticRegression(solver='liblinear')
model4.fit(X_train, Y_train)
predicted = model4.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print("4. Matrice de confusion: \n", matrix)


#----------------------------------------------------------------------------
# 5. CLASSIFICATION REPORT

test_size2 = 0.33
seed2 = 7
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=test_size2,random_state=seed2)
model5 = LogisticRegression(solver='liblinear')
model5.fit(X_train2, Y_train2)
predicted2 = model5.predict(X_test2)
report = classification_report(Y_test2, predicted2)
print("5. Report :",report)


#================================================================================
#================================================================================

# CALCUL DE L'ERREUR QUADRATIQUE (sert à donner la précision d'un estimateur)

import requests, csv, numpy as np
from numpy import *

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

# r = requests.get(url)

# with open('BostonHouse.csv','w') as f:
#     writer = csv.writer(f)
#     for line in r.iter_lines():
#         writer.writerow(line.decode('utf-8').split(','))

raw_data = open('BostonHouse.csv','rt')
data = np.genfromtxt('BostonHouse.csv', dtype=float, delimiter=",", names=True)

# recuperation du header du csv
dt = pandas.read_csv('BostonHouse.csv')
titre = []
for col in dt.columns:
    titre.append(col)

print(titre)
# recuperation des données csv avec pandas et header
df = pandas.read_csv('BostonHouse.csv', header=1, index_col=False, names=titre)

#------------------------------------------------------------------------------------

# 1. THE MEAN ABSOLUTE ERROR (somme des differences absolue entre prediction et valeur exacte)

filename2 = 'BostonHouse.csv'
names2 = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']

dataframe2 = read_csv(filename2, delim_whitespace=True, names=names2)
array2 = df.values
X2 = array2[:,0:13]
Y2 = array2[:,13]
kfold4 = KFold(n_splits=10, random_state=7)
model6 = LinearRegression()
scoring3 = 'neg_mean_absolute_error'
results4 = cross_val_score(model6, X2, Y2, cv=kfold4, scoring=scoring3)
print("1. MAE : %.3f (%.3f)" % (results4.mean(), results4.std()))

#-----------------------------------------------------------------------------------
# 2. MEAN SQARE ERROR (la flème)
#-----------------------------------------------------------------------------------

# 3. R² METRIC ( fournit une indication de la qualité d'un ensemble 
#                           de prédictions par rapport aux valeurs réelles)
#   --> output entre 0 et 1 (proche de 1 = bonne qualité)
