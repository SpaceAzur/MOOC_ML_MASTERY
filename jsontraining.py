import json
from pandas import *

json_data = '{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}'

print("\nen prog_imperative\n",type(json.loads(json_data)), json.loads(json_data))

loaded_json = json.loads(json_data)

for i in loaded_json:
    print("%s: %d" % (i, loaded_json[i]))


# EN PROG_OBJET ------------------------------------------------
class Test(object):
    def __init__(self,data):
        self.__dict__= json.loads(data)

test1 = Test(json_data)
print("\nen prog_objet\n",type(test1), test1.__dict__)
#---------------------------------------------------------------

with open('distros.json','r') as f:
    distros_dict = json.load(f)

for distro in distros_dict:
    print(distro['Name'], distro['Version'])

# avec PANDAS library
filename = 'distros.json'
pdata = read_json(filename)
print(pdata)