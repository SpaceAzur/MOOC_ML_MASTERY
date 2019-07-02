# METHODE DE RE-ECHANTILLONAGE => evaluer la performance de notre algo machine learning
#  --> Pour voir la performance de notre algo sur des données inconnues 
#  --> objectif : tester/prédire la pertinence de notre algo-ML sur des nouveau jeu de données avec un modèle d'évaluation

# 4 techniques différentes

#--------------------------------------------------------------------------------
# 1. TRAIN AND TEST-SETS (recommendé pour algo qui sont lent, avec un très large jeu de données)
#  --> Tester notre algo sur 67% des données (chronologique) 
#  --> Prédire les 33% des données restantes
#  --> Comparer les prédictions avec le vrai jeu de données (33%) restant

from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut, ShuffleSplit
from sklearn.linear_model import LogisticRegression

filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)

print("1. Accuracy : moyenne %.3f%%" % (result*100.0))

#-----------------------------------------------------------------------------------
# 2. K-FOLD CROSS-VALIDATION (recommendé en 1ere méthode et pour quand k = 3, 5 ou 10)
#    --> même idéé que 1. Pas sur 1/3 des données mais sur k sous-parties (donc k observations)
#    --> donc réputé plus précis que 1.

kfold = KFold(n_splits=10, random_state=7)
model2 = LogisticRegression(solver='liblinear')
results = cross_val_score(model2,X,Y, cv=kfold)
print("2. Accuracy : moyenne %.3f%% (ecart-type %.3f%%)" % (results.mean()*100.0, results.std()*100.0))

#------------------------------------------------------------------------------------
# 3. LEAVE ONE OUT CROSS-VALIDATION (recommendé pour travailler sur variance et ecart type)
#   --> idem que 2. mais k = 1 
#   --> considéré comme plus coûteux en calcul que 2.

loocv = LeaveOneOut()
model3 = LogisticRegression(solver='liblinear')
results3 = cross_val_score(model3, X, Y, cv=loocv)
print("3. Accuracy : %.3f%% (%.3f%%)" % (results3.mean()*100.0, results3.std()*100.0))
                    # nous voyons que la variance (42%) est très importante ...


#-----------------------------------------------------------------------------------
# 4. REPEATED RANDOM TEST-TRAIN SPLITS (recommendé pour travailler sur variance et ecart type)
#   --> la séparation en k sous-partie est randomisée et répété (ici 10 fois "n_splits2")
#   --> AVANTAGES : va aussi vite que 1. tout en réduisant la variance de 3.
#   --> INCONVENIENTS :  peut provoquer de la redondance a cause de la randomisation

n_splits2 = 10
test_size2 = 0.33
seed2 = 7
kfold2 = ShuffleSplit(n_splits=n_splits2, test_size=test_size2, random_state=seed2)
model4 = LogisticRegression(solver='liblinear')
results4 = cross_val_score(model4, X, Y, cv=kfold2)
print("4. Accuracy : %.3f%% (%.3f%%)" % (results4.mean()*100.0, results4.std()*100.0))
