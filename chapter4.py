# manage csv file into numpy.array

import requests, csv, numpy as np, pandas
from numpy import *
from pandas import *

# RECUPERATION DU CSV ET SAUVEGARDE => OK---------------------------------------------------------------------------
# r = requests.get("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")

# with open('out.csv', 'w') as f:
#     writer = csv.writer(f)
#     for line in r.iter_lines():
#         writer.writerow(line.decode('utf-8').split(','))----------------------------------------------------------

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# METHODE 1 - avec csv
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename,'rt')
reader = csv.reader(raw_data, delimiter=",", quoting=csv.QUOTE_NONE, skipinitialspace=True)
x = list(reader)
data = np.array(x).astype(float)
print("avec csv ",data.shape)

# METHODE 2 - avec numpy
raw_data2 = open(filename,'rt')
data2 = loadtxt(raw_data2, delimiter=",")
print("avec numpy ",data2.shape)

# METHODE 3 - avec pandas - directos depuis l'URL
data3 = read_csv(url, header=None, index_col=False)
print("avec pandas ",data3.shape)
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']







