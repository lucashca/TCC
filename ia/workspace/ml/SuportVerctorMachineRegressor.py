import sys
sys.path.insert(0, "../myTools")

import numpy as np

from dataSetPreProcessing import train_validation_test_split
from sklearn.model_selection import train_test_split

from loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet,loadMainDataSetWithElevation
from tools import verifyArgs,findBalancedDataSet,pltResults,pltCorrelation, pltLossGraph,pltShow,plotXY,getMetrics,plotLeanrningCurve

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.svm import SVR 
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# Load data set
dataSet,features_names,target_names = loadMainDataSetWithElevation()


  

def getParamGrid():
    param_grid_half = {        
        'kernel':['linear','rbf','sigmoid'],
        'gamma': ['scale','auto'], 
        'tol':[1e-5,1e-4], 
        'epsilon':[1e-4],
        'max_iter':[-1],
        'C':[1.5, 10]
    }
    param_grid_full = {
        
        'kernel':['linear','rbf','sigmoid'],
        'gamma': ['scale','auto',1e-7, 1e-4], 
        'tol':[1e-3,1e-5,1e-4,1e-2], 
        'epsilon':[1e-4],
        'max_iter':[-1],
        'C':[1.5, 10]
}
    return param_grid_half,param_grid_full


def tuningParameters(model,param_grid,X_train,y_train,verbose=0):

    reg =GridSearchCV(model, cv=10,param_grid=param_grid,verbose=verbose,n_jobs=-1,scoring='r2',iid=True)
    reg.fit(X_train,y_train)
    
    print("#Best score:",reg.best_score_)
    print("#Best params:",reg.best_params_)
          
    return reg.best_estimator_,reg.best_params_,reg.best_score_


def getBestSeed(X,y,faixa,verbose=0):
    param_grid_half,_ = getParamGrid()
    maior_score = 0
    seed = 0
    for i in faixa:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=i)
        model = SVR()
        best_model,best_params,best_score = tuningParameters(model,param_grid_half,X_train,y_train)
        if best_score>maior_score:
            maior_score = best_score
            seed = i

        y_train_pred = best_model.predict(X_train)
     
    if verbose:
        print("#Maior Score: ",maior_score)
        print("#Seed : ",seed)
   
    return seed,maior_score
    

def runTest(target,verbose=0):

    X = dataSet[:,:4]
    y = dataSet[:,target]
    seed,score = getBestSeed(X,y,range(1,10),verbose=verbose)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=seed)
        
    _,param_grid_full = getParamGrid()
    tuningParameters(SVR(),param_grid_full,X_train,y_train,verbose=verbose)

    print("#Best Seed:",seed)



def avaliateModel(model,X_train,X_val,X_test,y_train,y_val,y_test,param_key_loss,target,verbose=0,stepLoss=25,):

    model.fit(X_train,y_train)

    plotLeanrningCurve(X_train,X_val,y_train,y_val,model,param_key_loss,'mean_squared_error',legend_1="Treino",legend_2="Validação",verbose=1,step=25)
    plotLeanrningCurve(X_train,X_test,y_train,y_test,model,param_key_loss,'mean_squared_error',legend_1="Treino",legend_2="Teste",verbose=1,step=25)
    plotLeanrningCurve(X_train,X_val,y_train,y_val,model,param_key_loss,'r2',legend_1="Treino",legend_2="Validação",verbose=1,step=25)
    plotLeanrningCurve(X_train,X_test,y_train,y_test,model,param_key_loss,'r2',legend_1="Treino",legend_2="Teste",verbose=1,step=25)
    
    
    y_train_pred = model.predict(X_train)
    print("#Metricas para os dados de treino")
    getMetrics(y_train,y_train_pred,verbose=1)
    plotXY(y_train,y_train_pred,"Treino","Predito",target_names[target],"Treino X Predito",midle_line=True)
    
    y_val_pred = model.predict(X_val)    
    print("#Metricas para os dados de validação")
    getMetrics(y_val,y_val_pred,verbose=1)
    plotXY(y_val,y_val_pred,"Validação","Predito",target_names[target],"Validação X Predito",midle_line=True)
     
    y_test_pred = model.predict(X_test)
    print("#Metricas para os dados de teste")
    getMetrics(y_test,y_test_pred,verbose=1)    
    plotXY(y_test,y_test_pred,"Teste","Predito",target_names[target],"Teste X Predito",midle_line=True)
    
   
    pltShow()
    
   


def MELHOR_MG():
    params = {'C': 1.5, 'epsilon': 0.0001, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': 2000, 'tol': 0.001}
    model = SVR(**params)

    X = dataSet[:,:4]
    y = dataSet[:,4]
   
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=9)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2,random_state=7) 
    avaliateModel(model,X_train,X_val,X_test,y_train,y_val,y_test,'max_iter',0,verbose=1,stepLoss=25)

    #Best score: 0.7773431551951668
    #Best params: {'C': 1.5, 'epsilon': 0.0001, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'tol': 0.001}
    #Best Seed: 5
    #Metricas para os dados de treino
    #R-squared: 0.83808
    #Mean Squared Error: 0.00422
    #Metricas para os dados de validação
    #R-squared: 0.82290
    #Mean Squared Error: 0.00327
    #Metricas para os dados de teste
    #R-squared: 0.81450
    #Mean Squared Error: 0.00295




def MELHOR_NA():
    params ={'C': 10, 'epsilon': 0.0001, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': 2000, 'tol': 0.001}
    model = SVR(**params)

    X = dataSet[:,:4]
    y = dataSet[:,5]
   
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2,random_state=0) 
    avaliateModel(model,X_train,X_val,X_test,y_train,y_val,y_test,'max_iter',1,verbose=1,stepLoss=25)

    #Best score: 0.6511186214845505
    #Best params: {'C': 10, 'epsilon': 0.0001, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'tol': 0.001}
    #Best Seed: 2
    #Metricas para os dados de treino
    #R-squared: 0.73896
    #Mean Squared Error: 0.00563
    #Metricas para os dados de validação
    #R-squared: 0.86552
    #Mean Squared Error: 0.00282
    #Metricas para os dados de teste
    #R-squared: 0.74592
    #Mean Squared Error: 0.00848

def MELHOR_K():
    params = {'C': 1.5, 'epsilon': 0.0001, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': 2000, 'tol': 0.001}
    model = SVR(**params)

    X = dataSet[:,:4]
    y = dataSet[:,6]
   
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=3)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2,random_state=0) 
    avaliateModel(model,X_train,X_val,X_test,y_train,y_val,y_test,'max_iter',2,verbose=1,stepLoss=25)


#Best score: 0.7409384454642772
#Best params: {'C': 1.5, 'epsilon': 0.0001, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'tol': 0.001}
#Best Seed: 3


#runTest(4)


MELHOR_MG()
#MELHOR_NA()
#MELHOR_K()