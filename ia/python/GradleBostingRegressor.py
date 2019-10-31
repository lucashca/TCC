from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,max_error,explained_variance_score,median_absolute_error,mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from machineLearnig import createProcessByBestMean,createProcessByBestRes
import matplotlib.pyplot as plt

from dataSetUtils import normalizeColumn, deNormalizeColumn, convertCSV
from mpl_toolkits.mplot3d import Axes3D 

import pandas as pd
import numpy as np

def testAndPlot(l,e,d,r,m,p,rd,yColunm,dataSet):
    model = GradientBoostingRegressor(presort=p,loss=l,n_estimators=e,max_features=m,random_state=r,max_depth=d)
    x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=rd)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    mae = median_absolute_error(y_test,y_pred)
    maxError = max_error(y_test,y_pred)


    fig = plt.subplot(221)
    fig.scatter(x_test[:,0],y_test,c=y_test ,cmap="coolwarm")
    fig.set_title("Dados de Teste R2 Score: %.5f Mean Square Error: %.5f Mean Absolute Error: %.5f Max Error: %.5f"%(r2,mse,mae,maxError))
    fig.set_ylabel('Ca')
    fig.set_xlabel('Latitude')
    fig.set_ylim(0, 1)

    fig = plt.subplot(222)
    fig.scatter(x_test[:,1],y_test,c=y_test ,cmap="coolwarm")
    fig.set_title("Dados de Teste")
    fig.set_ylabel('Ca')
    fig.set_xlabel('Longitude')
    fig.set_ylim(0, 1)


    fig = plt.subplot(223)
    fig.scatter(x_test[:,0],y_pred,c=y_pred ,cmap="coolwarm")
    fig.set_title("Dados Preditos")
    fig.set_ylabel('Ca')
    fig.set_xlabel('Latitude')
    fig.set_ylim(0, 1)
    fig = plt.subplot(224)
    fig.scatter(x_test[:,1],y_pred,c=y_pred ,cmap="coolwarm")
    fig.set_title("Dados Preditos")
    fig.set_ylabel('Ca')
    fig.set_xlabel('Longitude')
    fig.set_ylim(0, 1)
    
    plt.show()

def testConfiguration(l,e,d,r,m,p,i,yColumn,dataSet):
    global yColunm
    model = GradientBoostingRegressor(presort=p,loss=l,n_estimators=e,max_features=m,random_state=r,max_depth=d)
    x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColumn],test_size=0.3,random_state=i)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    print("R2",r2)
    print("MSE",mse)
    
    return mse,r2



def testAndCompareDataFrame(l,e,d,r,m,p,rd,yColunm,dataSet,dataSetOriginal):
    model = GradientBoostingRegressor(presort=p,loss=l,n_estimators=e,max_features=m,random_state=r,max_depth=d)
    x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=rd)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    
    y_pred2 = deNormalizeColumn(dataSetOriginal,yColunm,y_pred)
    y_test2 =deNormalizeColumn(dataSetOriginal,yColunm,y_test)


    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred,'Original':y_test2,'Predicted De':y_pred2})
    df1 = df.head(20)
    print(df1)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    r22 = r2_score(y_test2,y_pred2)
    mse2 = mean_squared_error(y_test2,y_pred2)
    ev = explained_variance_score(y_test,y_pred)
    ev2 = explained_variance_score(y_test2,y_pred2)
    mae = median_absolute_error(y_test,y_pred)
    mae2 = median_absolute_error(y_test2,y_pred2)
    mle = mean_squared_log_error(y_test,y_pred)
    mle2 = mean_squared_log_error(y_test2,y_pred2)
    print("R2 Norm ",r2)
    print("MSE Norm ",mse)
    print("EVS Norm ",ev)
    print("MAE Norm ",mae)
    print("MLE Norm ",mle)

    print("R2 ",r22)
    print("MSE ",mse2)
    print("EVS ",ev2)
    print("MAE ",mae2)
    print("MLE ",mle2)
    
    	
def createRegressor(param_grids):
    regressor = []
    for l in param_grids["loss"]:
        for e in param_grids["n_estimators"]:
            for d in param_grids["max_depth"]:
                for r in param_grids["random_state"]:
                    for m in param_grids["max_features"]:
                        for p in param_grids["presort"]:
                            regressor.append([[l,e,d,r,m,p],GradientBoostingRegressor(presort=p,loss=l,n_estimators=e,max_features=m,random_state=r,max_depth=d)])
    return regressor



import sys

params = sys.argv[1:]
core = int(params[0],10)
yColumn = int(params[1],10)


scaler =  MinMaxScaler()

csvFile = pd.read_csv("mainDataSet1.csv",usecols=[1,2,3,4,5,6,7])
dataSetOriginal = np.array(csvFile.values)

#csvFile = convertCSV(csvFile.values)
dataSet = np.array(csvFile.values)



normalizeColumn(dataSet,0)
normalizeColumn(dataSet,1)
normalizeColumn(dataSet,yColumn)


d = [i for i in range(1,10)]
d.append(None)
e = range(1,10)



param_grids = {
            "loss"         : ["ls", "lad", "huber", "quantile"],
            "n_estimators" : e,
            "max_depth"    : d,
            "random_state" : [0], 
            "max_features" : ["log2","auto"],
            "presort"      : [True, False]
        }



regressor = createRegressor(param_grids)
if __name__ == '__main__':
    print("%d Modelos a serem processados"%(len(regressor)))
    createProcessByBestRes(core,regressor,dataSet,range(1,100),yColumn)
    #createProcessByBestMean(4,regressor,dataSet,range(64,74),yColumn)  
    #testAndCompareDataFrame(3, 1, 'log2', False, 0, 33,yColumn,dataSet,dataSetOriginal)
    #testAndPlot('lad', 100, 13, 0, 'log2',True, 87,yColumn,dataSet)
    
'''
Coluna 2 

[0.012156685550497428, 0.6266349283303412, ['ls', 9, 8, 0, 'auto', 48], ['ls', 9, 7, 0, 'auto', 87]]
[0.011893571625827782, 0.5930208655855519, ['huber', 9, None, 0, 'log2', 48], ['huber', 9, None, 0, 'auto', 87]]
Thread2 : 56.06 % 37/66 - MSE: 0.014 R2: 0.619 ['huber', 100, 12, 0, 'auto', 48] ['huber', 100, 15, 0, 'log2', 87]
Thread1 : 92.42 % 61/66 - MSE: 0.015 R2: 0.606 ['lad', 100, 16, 0, 'log2', 11] ['lad', 100, 13, 0, 'log2', 87]

Coluna 3
[0.004795091545767671, 0.6006897739455531, ['ls', 9, None, 0, 'log2', 33], ['ls', 9, None, 0, 'log2', 33]]
Thread2 : 8.18 % 36/440 - MSE: 0.005 R2: 0.553 ['huber', 10, 18, 0, 'log2', 48] ['huber', 10, 11, 0, 'log2', 33]
Thread1 : 22.05 % 97/440 - MSE: 0.005 R2: 0.510 ['lad', 11, 15, 0, 'log2', 17] ['lad', 11, 18, 0, 'log2', 33]
[0.004828851335528612, 0.5102662447402222, ['huber', 9, 4, 0, 'auto', 48], ['huber', 9, None, 0, 'auto', 33]]
[0.004235177164188011, 0.6473165246854946, ['ls', 18, 14, 0, 'log2', 33], ['ls', 18, 14, 0, 'log2', 33]]

Coluna 4

[0.010393120414772814, 0.5075599917713223, ['ls', 9, 7, 0, 'auto', 72], ['ls', 9, None, 0, 'log2', 44]]
[0.023281947076943643, 0.31799914003428686, ['quantile', 9, None, 0, 'log2', 13], ['quantile', 9, None, 0, 'auto', 69]]
[0.011918811944377989, 0.3865585094103414, ['lad', 6, None, 0, 'auto', 26], ['lad', 8, None, 0, 'auto', 72]]
[0.010651714379987832, 0.4997823881356702, ['huber', 6, None, 0, 'auto', 83], ['huber', 9, None, 0, 'log2', 33]]

'''