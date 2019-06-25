# quick and efficient data statistics understanding

import requests, csv, numpy as np, pandas
from numpy import *
from pandas import *

filename = "pima-indians-diabetes.data.csv"

# definition du header
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# lecture du fichier csv avec header
data = read_csv(filename, names=names)

# lecture des 20 premières lignes
peek = data.head(10)
print(peek)

# identifier le type de données de chaque colonne
types = data.dtypes
print(types)

# affiner la disposition des données (display)
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)

# Class distribution
class_counts = data.groupby(['class','age']).size()
print(class_counts)
