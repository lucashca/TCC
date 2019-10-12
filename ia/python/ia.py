import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import preprocessing
import sys
import threading
import time
from multiprocessing import Process, Lock



def fixCsv(arr):
    #Corrige o dataset, convertendo strig para float e também coverte a unidade dos nutriente de mg/l para cmolc/dm3
    for data in arr:
        data[0] = float(data[0])
        data[1] = float(data[1])
        data[2] = round(float(data[2])/200.4,2) #Ca
        data[3] = round(float(data[3])/121.56,2) #Mg
        data[4] = round(float(data[4])/230,2) #Na
        data[5] = round(float(data[5])/391,2) #K
        data[6] = round(float(data[6])/506.47,2) #Cl
        
    return arr

def getMaxValues(data):
    ''' Esta função é utilizada para obter os valores máximos de cada coluna do DataSet'''
    maxValues = []
    for i in range(data.shape[1]):
        maxValues.append(max(abs(data[:,i])))
    return maxValues


def getMinValues(data):
    ''' Esta função é utilizada para obter os valores mínimos de cada coluna do DataSet'''
    minValues = []
    for i in range(data.shape[1]):
        minValues.append(min(abs(data[:,i])))
    return minValues


def normalizeData(dataSet):
    ''' Esta função é utilizada para normalizar os dados em uma faixa entre 0 e 1 '''
    maximos = getMaxValues(dataSet)
    minimos = getMinValues(dataSet)
    data = np.zeros(dataSet.shape)
    
    for i in range(dataSet.shape[1]):
         data[:,i] = (abs(dataSet[:,i]) - minimos[i])/(maximos[i] - minimos[i])
        #data[:,i] = abs(dataSet[:,i]/maximos[i])
    return data


def run(regression,totalIter,name):

    model = regression[0][1]
    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)
    #score = r2_score(yTest,yPred)
    #maior = score
    menorError = mean_squared_error(yTest,yPred)
    bestModel = []
    atual = 0
    for i,reg in enumerate(regression):
        try:
            model = reg[1]
            model.fit(xTrain,yTrain)
            yPred = model.predict(xTest)
            #score = r2_score(yTest,yPred)
            erro =  mean_squared_error(yTest,yPred)
             
            if not totalIter % 2 == 0:
                totalIter = totalIter + 1

           
            
            atual = atual+1
            
            percent = (atual*100) / totalIter
            #print("%s %d/%d - %.2f / %.2f"%(name,atual, totalIter,percent,maior))
            print("%s %d/%d - %.2f / %.2f"%(name,atual, totalIter,percent,menorError))

            if erro < menorError:
            
                menorError = erro
                bestModel = reg
            

            '''
            if score > maior:
                maior = score
                bestModel = reg
            '''
        except:
            pass
    return bestModel


def mainWork(regression,totalIter,name):
    try:
        bestModel = run(regression,totalIter,name)    
        model = bestModel[1]
        model.fit(xTrain,yTrain)
        yPred = model.predict(xTest)
        score = r2_score(yTest,yPred)
        mse = mean_squared_error(yTest,yPred)

        ax = plt.subplot(221,projection='3d')
        ax.scatter(dataSetNormalizada[:,0],dataSetNormalizada[:,1],dataSetNormalizada[:,yColunm] ,c=dataSetNormalizada[:,yColunm] ,cmap="coolwarm")
        m = str(bestModel[0])
        ax.set_title("%.5f %s %s %.5f"% (mse,name,m,score))
        
        
        ax1 = plt.subplot(222,projection='3d')
        ax1.scatter(xTrain[:,0], xTrain[:,1],yTrain,c=yTrain ,cmap="coolwarm")
        ax1.set_title("Train Data")

        ax2 = plt.subplot(223,projection='3d')
        ax2.scatter(xTest[:,0], xTest[:,1],yTest,c=yTest ,cmap="coolwarm")
        ax2.set_title("Test Data")


        ax3 = plt.subplot(224,projection='3d')
        ax3.scatter(xTest[:,0], xTest[:,1],yPred,c=yPred ,cmap="coolwarm")
        ax3.set_title("Pred Data")

        plt.show()
    
    except:
        pass
    



## Import DataSet
csv = pd.read_csv("novdataset.csv",usecols=[1,2,3,4,5,6,7])
csv = fixCsv(csv.values)
dataSet = np.array(csv)


#csv = pd.read_csv("sela.csv",usecols=[0,1,2])
#dataSet = np.array(csv.values)


## Converte para uma matrix do numpy
## Normaliza o DataSet na faixa de 0 e 1
#dataSetNormalizada = normalizeData(dataSet)
#dataSetNormalizada = dataSet

dataSet[:,0] = preprocessing.normalize([dataSet[:,0]])
dataSet[:,1] = preprocessing.normalize([dataSet[:,1]])

dataSetNormalizada = dataSet

yColunm = 2

bestRandomState = [25,0,18,20,21,23,24,28,39]

xTrain,xTest,yTrain,yTest = train_test_split(dataSetNormalizada[:,:2],dataSetNormalizada[:,yColunm],test_size=0.3,random_state=25)
regression = []

ACTIVATION_TYPES = ["relu","logistic","tanh","identity"]
SOLVER_TYPES = ["lbfgs"]
ALPHA = [0.00001,0.001,0.01,0.1,1]
LEARNING_RATE_TYPES = ["constant","invscaling","adaptative"]


''''
for activation in ACTIVATION_TYPES:
    for solver in SOLVER_TYPES:
        for alfa in ALPHA:
            for lrt in LEARNING_RATE_TYPES:
                for r in range(100):
                    if lrt == "adaptative" and activation == "sgd":
                        regression.append([[activation,solver,alfa,lrt,r],MLPRegressor(hidden_layer_sizes=(10,10,10),activation=activation,solver=solver,alpha=alfa,learning_rate=lrt,max_iter=1000,random_state=r)])
                    elif not lrt == "adaptative" :
                        regression.append([[activation,solver,alfa,lrt,r],MLPRegressor(hidden_layer_sizes=(10,10,10),activation=activation,solver=solver,alpha=alfa,learning_rate=lrt,max_iter=1000,random_state=r)])
'''

for depth in range(10):
    for r in range(10):
        #regression.append([[depth,r],DecisionTreeRegressor(max_depth=depth,random_state=r)])   
        for s in range(10):
            regression.append([[depth,r,s],RandomForestRegressor(max_depth=depth+100, random_state=r,n_estimators=s)])
    print(depth)


 



totalWorkers = 5

worksize = int(len(regression)/totalWorkers) + 1
for i in range(totalWorkers):
    inicio = i*worksize
    fim = i*worksize + worksize
    if fim > len(regression):
        fim = len(regression)

    t = Process(target=mainWork,args=(regression[inicio:fim],worksize,"Thred %d" %i))
    t.start()














