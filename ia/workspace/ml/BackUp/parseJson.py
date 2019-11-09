
import pandas as pd
import json
import codecs
import numpy as np

data = pd.read_csv("datasets/DataSetWithElevationOriginal.csv",usecols=[0,1,2,3,4,5,6,7])


arr = np.array(data.values)

def removerLatLngDuplicaos(data):
    latitudes = data[:,1]
    longitudes = data[:,2]
    duplicados = []
    for i in range(len(latitudes)):
        lat = latitudes[i]
        lng = longitudes[i]
        cont = 0
        for j in range(len(longitudes)):
            lat2 = latitudes[j]
            lng2 = longitudes[j]
            if ((lat == lat2) and (lng == lng2)):
                
                if(cont > 0):
                    print(lat,lng,data[i,0])
                    print(lat2,lng2,data[j,0])
                
                    duplicados.append(data[i,0])
                cont+=1


removerLatLngDuplicaos(arr)

y = data.values.tolist()

file_path = "./dataset.json" ## your path variable
json.dump(y, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

