# PREPARE THE DATA FOR MACHINE LEARNING
import requests, csv, numpy as np, pandas
from numpy import set_printoptions
from numpy import *
from pandas import *
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# POUR INFO:  2 methodes pour preparer et transformer nos données
#  ** methode 1 Fit & multiple Transform (ajustement et transformation multiple)
#  ** methode 2 Combined Fit & Transform (ajustement et transformation combinées)


# RESCALE redimensionner les données (mettre data.valeurs entre 0 et 1) => pour meilleur digestion par ML
array = data.values

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
# scaler = MinMaxScaler(feature_range=(0,1))
# rescaledX = scaler.fit_transform(X)

# # summarize transformed data
# set_printoptions(precision=3)
# print(rescaledX[0:5,:])

# # standardize datas (Normalisation des données) 
# # => convertit les données en une distribution gaussienne (loi Normale centré réduite) StandardScaler
# Nscaler = StandardScaler().fit(X)
# NrescaledX = scaler.transform(X)
# set_printoptions(precision=3)
# print(NrescaledX[0:5,:])

# je teste donc si aucune valeur n'est supérieur à 1
# nparray = np.array(NrescaledX)
# for i, ligne in enumerate(nparray):
#     for j, value in enumerate(ligne):
#         if (value > 1):
#             print(value)

# # Normalize datas (lengh of 1)
# NNscaler = Normalizer().fit(X)
# NormalizedX = NNscaler.transform(X)
# set_printoptions(precision=3)
# print(NormalizedX[0:5,:])

# Binarize data
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
set_printoptions(precision=3)
print(binaryX[0:5,:])

