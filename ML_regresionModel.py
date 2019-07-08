import numpy as np
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NATURE DES DONNEES : Chaque ligne de données décrit un quartier de Boston ou une de ces villes de banlieue
# OBJECTIF           : PREDICTION DES PRIX DE L'IMMOBILIER DE BOSTON
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  CRIM     per capita crime rate by town                                         | CRIM    taux de criminalité par habitant par ville
#  ZN       proportion of residential land zoned for lots over 25,000 sq.ft.      | ZN      proportion de terrains résidentiels zonés pour des terrains de plus de 25 000 pi.ca.
#  INDUS    proportion of non-retail business acres per town                      | INDUS   proportion d'acres de commerces autres que de commerce de détail par ville
#  CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) | CHAS    Variable factice de Charles River (= 1 si le tracé est lié à la rivière; 0 sinon)
#  NOX      nitric oxides concentration (parts per 10 million)                    | NOX     concentration en oxydes d'azote (parties par 10 millions)
#  RM       average number of rooms per dwelling                                  | RM      nombre moyen de pièces par logement
#  AGE      proportion of owner-occupied units built prior to 1940                | AGE     proportion d'unités occupées par le propriétaire construites avant 1940
#  DIS      weighted distances to five Boston employment centres                  | DIS     distances pondérées à cinq centres d'emploi de Boston
#  RAD      index of accessibility to radial highways                             | RAD     indice d'accessibilité aux autoroutes radiales
#  TAX      full-value property-tax rate per $10,000                              | TAX     Taux de la taxe foncière de pleine valeur par tranche de 10 000 $
#  PTRATIO  pupil-teacher ratio by town                                           | PTRATIO ratio élèves / enseignant par ville
#  B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town        | B       1000 (Bk - 0.63) ^ 2 où Bk est la proportion de Noirs par ville
#  LSTAT    % lower status of the population                                      | LSTAT   % statut inférieur de la population
#  MEDV     Median value of owner-occupied homes in $1000's                       | MEDV    Valeur médiane des logements occupés par leurs propriétaires en milliers de dollars
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# chargement du fichier de données
fichier = 'Boston.csv'
entetes = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
donnees = read_csv(fichier) # par défault read_cv prend la 1ere ligne du fichier comme en-tête

# Je génère la description statistique de chaque attribut (compteur, moyenne, dispersion, min, max, etc)
set_option('precision', 1) # initialisation 
print(donnees.describe())  # affichage

# etude des correlations entre attributs
set_option('precision', 2)
print(donnees.corr(method='pearson'))


# visualisation des attributs | histogrammes
donnees.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()

# visualisation des attributs | courbe de densité (plus indiqué pour carateriser la distribution des attributs)
donnees.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=True, fontsize=1)
pyplot.show()

# visualisation des attributs | box and whisker plots (plus indiqué pour faire apparaitre les valeurs aberrantes)
donnees.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
pyplot.show()


# Visualisation mutli-modal => scatter plot matrix (matrice de dispersion) peu lisible et peu pertinent je trouve
# scatter_matrix(donnees)
# pyplot.show()

# Visualisation multi-modal => matrice de correlation
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(donnees.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(entetes)
ax.set_yticklabels(entetes)
pyplot.show()

'''
A terme, pour ne pas biaiser le modèle, nous devons trouver les attribus les plus corrélés entre eux et 
les enlever du modèle'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# separation des données en un jeu de test et un jeu de validation

tab = donnees.values
X = tab[:,0:13]
Y = tab[:,13]
taille_jeu_de_validation = 0.20
seed = 7
X_entrainement, X_validation, Y_entrainement, Y_validation = train_test_split(X,Y, test_size=taille_jeu_de_validation, random_state=seed)

# preparation du jeu de test (va permettre de dégager un algorithme qui collera le mieux aux données)
nb_scission = 10
seed = 7 # ex: dans +ProchesVoisins, le nb de voisins sera de 7
scoring = 'neg_mean_squared_error'

# nous allons tester 6 algorithmes de regression
modeles = []
modeles.append(('\nRegression Lineaire', LinearRegression()))
modeles.append(('LASSO', Lasso()))
modeles.append(('ElasticNet', ElasticNet()))
modeles.append(('+Proche Voisins', KNeighborsRegressor()))
modeles.append(('Arbre de Decision', DecisionTreeRegressor()))
modeles.append(('Vecteur de Support', SVR(gamma='auto')))


# Nous evaluons chaque modele
resultats = []
nom_modele = []
print("\nRESULTATS MSE( Moyenne des carrés des erreurs) :\n=> Que l'on cherche à minimiser")
for nom, modele in modeles:
    kfold = KFold(n_splits=nb_scission, random_state=seed)
    crossVal_resultat = cross_val_score(modele, X_entrainement, Y_entrainement, cv=kfold, scoring=scoring)
    resultats.append(crossVal_resultat)
    nom_modele.append(nom)
    message = "%s: %.3f (%.3f)" % (nom, crossVal_resultat.mean(), crossVal_resultat.std())
    print(message)

    '''fournit l'erreur quadratique moyenne de chaque algo (dit MSE), et sa variance
    Le MSE nous indique dans quelle mesure nos prédictions s'eloignent ou sont proches de la distribution existante
    Dans un schéma (abscisse/ordonnée) le MSE est la moyenne des carrés des écarts entre chaque points et la droite 
      des moindres carrés
    => NOUS RETIENDRONS LE MODELE QUI MINIMISE LE MSE : ici la regression linéaire (-21.3) et l'arbre de decision (-22.6)'''


# comparons les algorithmes graphiquement
fig = pyplot.figure()
fig.suptitle('Comparaison des algorithmes (MSE)')
ax = fig.add_subplot(111)
pyplot.boxplot(resultats)
ax.set_xticklabels(nom_modele)
pyplot.show()

''' PROBLEME : l'echelle de chaque attribut est peut être un frein a la bonne representation statistique 
et floutte les resultats. Notamment impact négativement la méthode des Vecteurs de Support et des +Proches Voisins'''

# Standardiser les données
'''=> lors de cette étape, nous allons ramener la distribution de tous les attributs dans une plage de valeur compris
entre 0 et 1. Elles auront ainsi la même échelle et donc la base de comparaison. Nous devrons faire attention a ne 
pas faire fuiter les données pendant ce processus de transformation.'''

# Eviter fuite = utiliser des 'Pipelines'
pipelines = []
pipelines.append(('S_Regression Lineaire', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('S_LASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('S_ElasticNet', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('S_+Proche Voisins', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('S_Arbre de decision', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
pipelines.append(('S_Vectuer de support', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

resultats2 = []
nom_modele2 = []
print("\nMSE après standardisation des attributs :\n")
for nome, modele in pipelines:
    kfold2 = KFold(n_splits=nb_scission, random_state=seed)
    crossVal_resultat2 = cross_val_score(modele, X_entrainement, Y_entrainement, cv=kfold2, scoring=scoring)
    resultats2.append(crossVal_resultat2)
    nom_modele2.append(nome)
    mesage = "%s: %.3f (%.3f)" % (nome, crossVal_resultat2.mean(), crossVal_resultat2.std())
    print(mesage)

    ''' Nous voyons que l'erreur quadratique a évolué. +ProcheVoisin devient le modele le plus interressant'''

# comparons de nouveau les algorithmes graphiquement
fig2 = pyplot.figure()
fig2.suptitle('Comparaison des algorithmes (MSE)')
ax = fig2.add_subplot(111)
pyplot.boxplot(resultats2)
ax.set_xticklabels(nom_modele2)
pyplot.show()

''' nous voyons que les +ProchesVoisins a la distribution la plus serrée et le + petit MSE => modele le + pertinent'''

# améliorons les resultats par règlages
''' testons d'autres valeurs pour 'seed' qui pourraient améliorer les resultats => prenons {1 <= seed <= 21}'''

scaler = StandardScaler().fit(X_entrainement)
rescaledX = scaler.transform(X_entrainement)
k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)

modele3 = KNeighborsRegressor()
kfold3 = KFold(n_splits=nb_scission, random_state=seed)
grid = GridSearchCV(estimator=modele3, param_grid=param_grid, scoring=scoring, cv=kfold3)
grid_resultat = grid.fit(rescaledX, Y_entrainement)

# affichage
print("Meilleur: %f utilisant %s" % (grid_resultat.best_score_, grid_resultat.best_params_))
means = grid_resultat.cv_results_['mean_test_score']
stds = grid_resultat.cv_results_['std_test_score']
params = grid_resultat.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) avec: %r" % (mean, stdev, param))


# autre methode d'amélioration: les ensembles (boosting et/ou bagging) toujours standardiser
ensembles = []
ensembles.append(('S_Boosting_AdaBoost', Pipeline([('Scaler', StandardScaler()),('AB',
AdaBoostRegressor())])))
ensembles.append(('S_Boosting_Gradient', Pipeline([('Scaler', StandardScaler()),('GBM',
GradientBoostingRegressor())])))
ensembles.append(('S_Bagging_RandomForest', Pipeline([('Scaler', StandardScaler()),('RF',
RandomForestRegressor(n_estimators=10))])))
ensembles.append(('S_Bagging_ExtraTrees', Pipeline([('Scaler', StandardScaler()),('ET',
ExtraTreesRegressor(n_estimators=10))])))

results = []
names = []
print("\nAmélioration par methode des ENSEMBLES, apres standardisation\n")
for name, model in ensembles:
    kfold = KFold(n_splits=nb_scission, random_state=seed)
    cv_results = cross_val_score(model, X_entrainement, Y_entrainement, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

    ''' La methode des ensembles Gradient donne de bons resultats: NOUS ALLONS DONC REGLER la methode des Gradient
    pour tenter d'améliorer encore les predictions. Pour l'instant, les paramètre de la methode des Gradient
    ont leurs valeurs par défault
    '''

# Reglages methodes des ensembles : Gradient
''' nous allons chercher la valeur optimale du paramètre 'n_estimators' de la méthode des Gradients
Sa valeur par default est de 100
Nous paramétrons sa valeur entre 50 et 550, par pas de 50
'''

scaler2 = StandardScaler().fit(X_entrainement)
rescaledX2 = scaler2.transform(X_entrainement)
param_grid2 = dict(n_estimators=np.array([50,100,150,200,250,300,350,400,450,500,550]))
modele4 = GradientBoostingRegressor(random_state=seed)
kfold4 = KFold(n_splits=nb_scission, random_state=seed)
grid2 = GridSearchCV(estimator=modele4, param_grid=param_grid2, scoring=scoring, cv=kfold4, iid=True)
grid_resultat2 = grid2.fit(rescaledX2, Y_entrainement)

print("\nMeilleur: %f utilisant %s" % (grid_resultat2.best_score_, grid_resultat2.best_params_))
means = grid_resultat2.cv_results_['mean_test_score']
stds = grid_resultat2.cv_results_['std_test_score']
params = grid_resultat2.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) avec: %r" % (mean, stdev, param))

    ''' nous approchons un resultat optimale avec un n_estimators = 500 
    Nous pouvons désormais finaliser le modele '''


# preparer le modele
scaler3 = StandardScaler().fit(X_entrainement)
rescaledX3 = scaler3.transform(X_entrainement)
modele_final = GradientBoostingRegressor(random_state=seed, n_estimators=500)
modele_final.fit(rescaledX3, Y_entrainement)

# transformer le jeu de données de validation
rescaledValidationX = scaler3.transform(X_validation)
predictions = modele_final.predict(rescaledValidationX)
print("\nMSE optimal par +Proche_Voisin, affiné par méthodes des Ensemble Gradient\n", mean_squared_error(Y_validation, predictions))


