import sys
from minisom import MiniSom
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec




def parseInt(var):
    if(var % 1 != 0):
        var = round(var)+1

    var = int(var)
    return var


i = 0

maxEpochs = 100
somSize = 10
__dir='./datasets/'
filename='var.csv'
try:
    arg = sys.argv[1:]
    
    if(arg[0]):
        maxEpochs = int(arg[0])
    if(arg[1]):
        somSize = int(arg[1])
    if(arg[2]):
        print(arg[2])
        filename = arg[2]
except :
    pass




__outDir = './out/'
data = np.genfromtxt(__dir+filename, delimiter=',')
# Elimine header
data = data[1:]
dataColuns = 1
try:
    dataColuns = len(data[0])
except:
    pass
print('Linhas: ',len(data), ' Colunas: ',dataColuns)



def nomalizeData(d):
    d = d
    maxValue = max(d)
  
    d = d/(maxValue)
    return d

def createCsv(arr,name):
    global rowNumber
    csv_file = open(__outDir+name+'.csv','w')
    csv_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for d in arr:
        csv_file.writerow(d)

print('Normalizando dados')
data = np.apply_along_axis(nomalizeData,0,data)

createCsv(data,'dadosNormalizados')





# Initialization and training
som = MiniSom(somSize, somSize, dataColuns, sigma=1, learning_rate=0.5, 
              neighborhood_function='gaussian')

som.pca_weights_init(data)
print("Training...")
som.train_batch(data, maxEpochs, verbose=True)  # random training
print("\n...ready!")



plt.figure(figsize=(7, 7))
# Plotting the response for each pattern in the iris dataset

plt.subplot(2,1,1)
plt.pcolor(som.distance_map())  # plotting the distance map as background
plt.title('Matriz - U')
plt.colorbar()

plt.subplot(2,1,2)
frequencies = np.zeros((somSize, somSize))
for position, values in som.win_map(data).items():
    frequencies[position[0], position[1]] = len(values)
plt.title('Densidade de Dados')
plt.pcolor(frequencies)
plt.colorbar()

### Imprime mapa em relação aos pesos das colunas
'''
headers = 'Nitrogênio Foliar (g/kg),Fósforo(P) (mg/dm³),Potássio(K) (cmol c/dm³),Boro(B) (mg/dm³),Cálcio(Ca) (cmol c/dm³),Magnésio(Mg) (cmol c/dm³),Alumínio(Al) (cmol c/dm³),Sódio(Na) (cmol c/dm³),H+Al (cmol c/dm³),Zinco(Zn) (mg/dm³),Cobre(Cu) (mg/dm³),Manganês(Mn) (mg/dm³),PRNT do calcário'
feature_names = headers.split(',')

figureSize = dataColuns
figureSize = figureSize/3
figureSize = parseInt(figureSize)

W = som.get_weights()
plt.figure(figsize=(10, 10))
for i, f in enumerate(feature_names):
    plt.subplot(figureSize, 3, i+1)
    plt.title(f)
    plt.pcolor(W[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(somSize+1))
    plt.yticks(np.arange(somSize+1))
plt.tight_layout()
'''


plt.show()



