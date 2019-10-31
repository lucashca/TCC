from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,max_error
from sklearn.preprocessing import MinMaxScaler
from machineLearnig import createProcessByBestMean,createProcessByBestRes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import pandas as pd


def testAndPlot(d,e,m,b,r,rd,yColunm,dataSet):
    model = RandomForestRegressor(max_depth=d,n_estimators=e,max_features=m,bootstrap=b,random_state=r)
    x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=rd)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)

    fig = plt.subplot(211)
    fig.scatter(x_test[:,0],y_test,c=y_test ,cmap="coolwarm")
    fig.set_title("Dados de Teste R2: %.5f MSE: %.5f"%(r2,mse))
    fig.set_ylabel('Ca')
    fig.set_xlabel('Latitude')
    fig.set_ylim(0, 1)
    fig = plt.subplot(212)
    fig.scatter(x_test[:,0],y_pred,c=y_pred ,cmap="coolwarm")
    fig.set_title("Dados de Pred")
    fig.set_ylabel('Ca')
    fig.set_xlabel('Latitude')
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

def createRegressor(param_grids):
    regressor = []
    for d in param_grids["max_depth"]:
        for e in param_grids["n_estimators"]:
            for m in param_grids["max_features"]:
                for b in param_grids["bootstrap"]:
                    regressor.append([[d,e,m,b,0],RandomForestRegressor(max_depth=d,n_estimators=e,max_features=m,bootstrap=b,random_state=0)])
    return regressor

scaler =  MinMaxScaler()

csvFile = pd.read_csv("melhorDataSet.csv",usecols=[1,2,3,4,5,6,7])
yColumn = 2
dataSet = scaler.fit_transform(csvFile)

d = [i for i in range(1,100)]
d.append(None)
e = range(10,20)
param_grids = {
            "max_depth"    : d,
            "n_estimators" : e,
            "max_features" : ["log2","auto"],
            "bootstrap"    : [True, False],
        }


regressor = createRegressor(param_grids)
if __name__ == '__main__':
    createProcessByBestRes(4,regressor,dataSet,range(1,100),yColumn)
    #createProcessByBestMean(4,regressor,dataSet,range(64,74),yColumn)  
    #testAndPlot(9, 7, 'log2', False, 0, 38,yColumn,dataSet)


'''
[0.013043121834225452, 0.6744215196484691, [10, 6, 'log2', True, 0, 62], [9, 6, 'log2', False, 0, 64]]
[0.01311765269343221, 0.6069797481815808, [26, 8, 'log2', True, 0, 62], [26, 8, 'auto', True, 0, 64]]
[0.01311765269343221, 0.6069797481815808, [51, 8, 'log2', True, 0, 62], [51, 8, 'auto', True, 0, 64]]
[0.01311765269343221, 0.6069797481815808, [76, 8, 'log2', True, 0, 62], [76, 8, 'auto', True, 0, 64]]
'''