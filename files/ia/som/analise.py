import sys
sys.path.insert(0, '../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import math,sys

from minisom import MiniSom
from sklearn.preprocessing import minmax_scale, scale



def parseInt(var):
    if(var % 1 != 0):
        var = round(var)+1

    var = int(var)
    return var





__dir = './datasets/'
__outDir = './out/'


arg = sys.argv[1]

### Paramters for som
neighborhood_function = 'gaussian'
maxEpochs = 100
learning_rate = 0.5

if(arg):
    maxEpochs = int(arg)


numCols = 13
labelCols = 0
dataCols = numCols - labelCols
columns = list(range(labelCols,numCols))

#*******************************************

soloCsv = pd.read_csv(__dir+'solo.csv')
headers = 'Nitrogênio Foliar (g/kg),Fósforo(P) (mg/dm³),Potássio(K) (cmol c/dm³),Boro(B) (mg/dm³),Cálcio(Ca) (cmol c/dm³),Magnésio(Mg) (cmol c/dm³),Alumínio(Al) (cmol c/dm³),Sódio(Na) (cmol c/dm³),H+Al (cmol c/dm³),Zinco(Zn) (mg/dm³),Cobre(Cu) (mg/dm³),Manganês(Mn) (mg/dm³),PRNT do calcário'
feature_names = headers.split(',')
data = soloCsv[feature_names].values
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x),0,data)

#********************************************


somSize = 5*math.sqrt(len(data))
somSize = parseInt(somSize)
somAxis = math.sqrt(somSize)
somAxis = parseInt(somAxis)

size = somAxis = 10

print(size)
som = MiniSom(size, size, len(data[0]),neighborhood_function=neighborhood_function, sigma=1,learning_rate=0.5,random_seed=1)

som.random_weights_init(data)


print("Training...")
som.train_batch(data,maxEpochs,True) # training with 100 iterations
print("\n...ready!")



figureSize = len(feature_names)

figureSize = figureSize/3
figureSize = parseInt(figureSize)

W = som.get_weights()
plt.figure(figsize=(10, 10))
for i, f in enumerate(feature_names):
    plt.subplot(figureSize, 3, i+1)
    plt.title(f)
    plt.pcolor(W[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(size+1))
    plt.yticks(np.arange(size+1))
plt.tight_layout()
plt.show()



