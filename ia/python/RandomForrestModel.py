from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,max_error,explained_variance_score,median_absolute_error,mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from machineLearnig import createProcessByBestMean,createProcessByBestRes
import matplotlib.pyplot as plt

from dataSetUtils import normalizeColumn, deNormalizeColumn, convertCSV
from mpl_toolkits.mplot3d import Axes3D 

import pandas as pd
import numpy as np

def testAndPlot(d,e,m,b,r,rd,yColunm,dataSet):
    model = RandomForestRegressor(max_depth=d,n_estimators=e,max_features=m,bootstrap=b,random_state=r)
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
    fig.set_ylabel('K')
    fig.set_xlabel('Latitude')
    fig.set_ylim(0, 1)

    fig = plt.subplot(222)
    fig.scatter(x_test[:,1],y_test,c=y_test ,cmap="coolwarm")
    fig.set_title("Dados de Teste")
    fig.set_ylabel('K')
    fig.set_xlabel('Longitude')
    fig.set_ylim(0, 1)


    fig = plt.subplot(223)
    fig.scatter(x_test[:,0],y_pred,c=y_pred ,cmap="coolwarm")
    fig.set_title("Dados Preditos")
    fig.set_ylabel('K')
    fig.set_xlabel('Latitude')
    fig.set_ylim(0, 1)
    fig = plt.subplot(224)
    fig.scatter(x_test[:,1],y_pred,c=y_pred ,cmap="coolwarm")
    fig.set_title("Dados Preditos")
    fig.set_ylabel('K')
    fig.set_xlabel('Longitude')
    fig.set_ylim(0, 1)
    
    plt.show()

def testConfiguration(d,s,m,b,r,i,yColumn,dataSet):
    global yColunm
    model = RandomForestRegressor(max_depth=d,n_estimators=s,max_features=m,bootstrap=b,random_state=r)
    x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColumn],test_size=0.3,random_state=i)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    print("R2",r2)
    print("MSE",mse)
    
    return mse,r2

def testAndCompareDataFrame(d,e,m,b,r,rd,yColunm,dataSet,dataSetOriginal):
    model = RandomForestRegressor(max_depth=d,n_estimators=e,max_features=m,bootstrap=b,random_state=r)
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
    for d in param_grids["max_depth"]:
        for e in param_grids["n_estimators"]:
            for m in param_grids["max_features"]:
                for b in param_grids["bootstrap"]:
                    for r in param_grids["random_state"]:
                        regressor.append([[d,e,m,b,r],RandomForestRegressor(max_depth=d,n_estimators=e,max_features=m,random_state=r,bootstrap=b,criterion="mse")])
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
            "max_depth"    : d,
            "n_estimators" : e,
            "max_features" : ["auto","log2"],
            "bootstrap"    : [False,True], 
            "random_state" : range(0,10)
        }



regressor = createRegressor(param_grids)
if __name__ == '__main__':
    print("Regressores %d"%(len(regressor)))
    #createProcessByBestRes(core,regressor,dataSet,range(100,1000),yColumn)
    #createProcessByBestMean(4,regressor,dataSet,range(64,74),yColumn)  
    #testAndCompareDataFrame(3, 1, 'log2', False, 0, 33,yColumn,dataSet,dataSetOriginal)
    testAndPlot(None, 1, 'log2', False, 0, 950,yColumn,dataSet)
    

#Coluna 2
#[0.013133266970181141, 0.7005473636414423, [7, 2, 'log2', True, 5, 930], [7, 1, 'log2', True, 0, 950]]
#[0.013888210202454431, 0.7159839121933842, [8, 4, 'log2', False, 9, 950], [8, 4, 'log2', False, 9, 950]]
#[0.011767574788571737, 0.7303697855748534, [None, 3, 'log2', False, 7, 968], [None, 1, 'log2', False, 0, 950]]
#[0.011922694048873191, 0.699395946221422, [7, 9, 'log2', False, 0, 295], [8, 3, 'log2', False, 7, 840]]

# Coluna 3 

#[0.003747330481584376, 0.6879418531596317, [3, 1, 'log2', False, 0, 33], [3, 1, 'log2', False, 0, 33]]
#[0.0031251506204478126, 0.7397537484332997, [6, 2, 'log2', True, 6, 33], [6, 2, 'log2', True, 6, 33]]
#[0.003356718949938022, 0.7204699451704816, [4, 2, 'log2', True, 6, 33], [4, 2, 'log2', True, 6, 33]]
#[0.0036250619162775254, 0.6966080753395689, [9, 3, 'log2', True, 7, 31], [9, 2, 'log2', False, 2, 33]]

#Coluna 4
#[0.006510322095758903, 0.6921877952802179, [6, 2, 'auto', True, 5, 765], [8, 1, 'log2', True, 5, 731]]
#[0.006741072629439677, 0.6673644559337264, [None, 7, 'log2', True, 5, 765], [9, 2, 'log2', True, 5, 575]]
#[0.007580607424597495, 0.6215230480894705, [8, 5, 'log2', True, 0, 72], [None, 4, 'auto', True, 7, 8]]
#[0.007824224158694333, 0.5978032616861133, [8, 4, 'log2', True, 0, 72], [8, 4, 'log2', True, 0, 72]]

#[0.009228008127988734, 0.5343468940659876, [5, 5, 'auto', True, 0, 72], [5, 1, 'auto', True, 6, 53]]
#[0.00750129108733994, 0.564397097902015, [5, 2, 'auto', True, 5, 765], [5, 1, 'auto', True, 1, 888]]
#[0.008175638296361353, 0.45824708075321896, [3, 3, 'log2', True, 5, 681], [3, 3, 'auto', True, 4, 366]]

#coluna 5
#[0.0011979060055495338, 0.6162737423122828, [7, 4, 'auto', True, 7, 423], [7, 4, 'auto', True, 0, 575]]
#[0.0013254327476096813, 0.6048684295996806, [4, 9, 'log2', True, 3, 423], [5, 3, 'auto', True, 0, 720]]
#[0.0013599869300447822, 0.5857823125706259, [8, 8, 'log2', True, 8, 423], [None, 4, 'log2', True, 8, 611]]
#[0.0015044174105518294, 0.571456367126182, [5, 9, 'log2', False, 6, 64], [5, 4, 'log2', True, 0, 99]]

#[0.0020739237351495524, 0.38996821600828546, [3, 2, 'log2', True, 6, 64], [3, 2, 'auto', True, 0, 20]]
#[0.001443785538005866, 0.5548973858909094, [6, 6, 'log2', False, 4, 64], [6, 1, 'log2', True, 8, 50]]
#[0.001566701730812408, 0.5436295033806849, [8, 6, 'log2', True, 3, 64], [None, 1, 'log2', True, 8, 50]]
#[0.001418259220670665, 0.42420097065914897, [3, 4, 'log2', True, 7, 423], [3, 1, 'auto', True, 2, 659]]
