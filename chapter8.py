# CHOISIR LES FONCTIONNALITES permet d'accroitre la precision des prediction ou du modele, 
# en reduiant le bruit provoquer par des jeu de données non pertinent pour notre analyse

from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# # feature extraction
# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(X,Y)
# # summarize scores
# set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(X)
# # summarize selected features
# print(features[0:5,:])
        # ==> rechercher manuellement le nom des attributs avec leur indice
                        # [[148.    0.   33.6  50. ]
                        #  [ 85.    0.   26.6  31. ]
                        #  [183.    0.   23.3  32. ]
                        #  [ 89.   94.   28.1  21. ]
                        #  [137.  168.   43.1  33. ]]

# Recursion Feature Elimination (RFE)
        # ==> elimine recursivement les attributs non pertinent et construit un modele avec les attributs restant

# model = LogisticRegression(solver='liblinear')
# rfe = RFE(model,3)
# fit = rfe.fit(X,Y)
# print("Num Features : %d" % fit.n_features_)
# print("Selected features: %s" % fit.support_)
# print("Feature Ranking: %s" % fit.ranking_) # les meilleurs attributs = True


# Principal Component Analysis (PCA) 
    # ==> permet de réduire la taille des données ! :)   (compression)

# features extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance : %s" % fit.explained_variance_ratio_)
print(fit.components_)