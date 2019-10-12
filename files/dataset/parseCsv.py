
import pandas as pd
import json
import codecs

data = pd.read_csv("novdataset.csv",usecols=[0,1,2,3,4,5,6,7])

print(data.values)


y = data.values.tolist()

file_path = "./dataset.json" ## your path variable
json.dump(y, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

