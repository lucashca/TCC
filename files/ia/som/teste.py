import numpy as np
import math
from minisom import MiniSom
import csv
from pylab import plot,axis,show,pcolor,colorbar,bone
import scipy.misc
from scipy import ndimage
import sys


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

## **************************************************
def normalizeArray(x):
    return (x/np.linalg.norm(x))


def createCsv(arr,name):
    global rowNumber
    csv_file = open(__outDir+name+'.csv','w')
    csv_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for d in arr:
        csv_file.writerow(d)


def createClassName(x,y):
    className = {}
    cont = 0
    for i in range(x):
        for j in range(y):
            cont = cont + 1
            key = (i,j)
            value = str(cont)
            className.__setitem__(key,value)
    return className


def parseArrayDataToInt(arr = []):
    newArr = []
    for i in arr:
        line = []
        for j in i:
            data = []
            for d in j:
                data.append(int(d))
            line.append(data)
        newArr.append(line)
    return newArr

def parseInt(var):
    if(var % 1 != 0):
        var = round(var)+1

    var = int(var)
    return var




##***************************************************************



__dir = './datasets/'
__outDir = './out/'

### Paramters for som
neighborhood_function = 'gaussian'
maxEpochs = 100
learning_rate = 0.5


arg = False
try:
    arg = sys.argv[1]
    if(arg):
        maxEpochs = int(arg)
except :
    pass



### Data Atributes
numCols = 13
labelCols = 0
dataCols = numCols - labelCols
columns = list(range(labelCols,numCols))




data = np.genfromtxt(__dir+'solo.csv', delimiter=',',usecols=columns) 
data = data[1:]
dataOriginal = np.copy(data)

##Normalizando dados
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x),0,data)


## Default SOM
somAxis = 3
somSize = 9

## Calculando SomSize
if(len(data)>50):
    somSize = 5*math.sqrt(len(data))
    somSize = parseInt(somSize)
    somAxis = math.sqrt(somSize)
    somAxis = parseInt(somAxis)
    

somX = somAxis = 30
somY = somAxis
print(somSize,' - ',somY,'x',somX)


className = createClassName(somX,somY)


som = MiniSom(somX,somY,dataCols,sigma=2.5,learning_rate=learning_rate,neighborhood_function=neighborhood_function)
som.random_weights_init(data)
starting_weights = som.get_weights().copy()  # saving the starting weights

print("Training...")
som.train_batch(data,maxEpochs,True) # training with 100 iterations
print("\n...ready!")



dataOut = []
csv_file = open(__dir+'solo.csv','r')
l = csv_file.readline()  
arr = l.split(',')
arr.append('Classe')
dataOut.append(arr)
classifiedValues = {}

for d in range(len(data)):
    
    w = som.winner(data[d])
    dataOut.append(np.append(dataOriginal[d],className.get(w)))

    v = classifiedValues.get(className.get(w))   
    if(v):
        v = v+1
        classifiedValues.__setitem__(className.get(w),v)
    else:
        classifiedValues.__setitem__(className.get(w),1)
          



list_key_value = [ [k,v] for k, v in classifiedValues.items() ]
list_key_value.insert(0,['class','lenght'])
createCsv(dataOut,'soloOut')
createCsv(list_key_value,'class')








# show the result

plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.bone()
plt.title('Mapa de Kohonem')
plt.pcolor(som.distance_map().T)
plt.colorbar()
plt.tight_layout()



#Quanto mais escuro mais denso é a posição
plt.subplot(222)
plt.title('Densidade de Dados')
frequencies = np.zeros((somX,somY))
d = som.win_map(data)
for position in d:
    frequencies[position[0], position[1]] = len(d.get(position))

plt.pcolor(frequencies, cmap='Blues')
plt.colorbar()


final_weights = som.get_weights()
rotated_img = ndimage.rotate(starting_weights, 90)
rotated_img2 = ndimage.rotate(final_weights, 90)




plt.subplot(223)
plt.title('initial colors')
plt.pcolor(rotated_img[0])
plt.colorbar()


plt.subplot(224)
plt.title('learned colors')
plt.pcolor(rotated_img2[0])
plt.colorbar()

plt.tight_layout()
plt.show()


