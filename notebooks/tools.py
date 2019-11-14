
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np

def verifyArgs(args):
   
    try:
        y_column = int(args[0],10)    
    except:
        y_column = 2
    try:
        random_state = int(args[1],10)    
    except:
        random_state = 0

    return y_column,random_state


def findBalancedDataSet(faixa,X,y,searchMethod,negRefit=False,verbose=0):
    print("Buscando os melhores dados para o treino")
    global_best_score = 0
    global_best_seed = 0
    global_best_model = None
    global_best_params = {}

    total = len(faixa)
    cont = 0

    operador = 1
    if negRefit:
        operador = -1

    for r in faixa:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=r)
        best_model,best_params,best_score =  searchMethod(X_train,y_train,verbose=verbose)
        
        if best_score*operador > global_best_score:
            global_best_score = best_score
            global_best_seed = r
            global_best_params = best_params
            global_best_model = best_model
        cont+=1


        print("%d / %d : Best Score: %.5f "%(cont,total,global_best_score))
    print()
    print("#Conluido - Best Score: %.5f Semente: %d "%(global_best_score,global_best_seed))
    return global_best_model,global_best_params,global_best_score,global_best_seed



def pltResults(best_model,x_index,y_index,X_train,X_test,y_train,y_test,feature_names,target_names):
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    r2_test = r2_score(y_test,y_test_pred)
    r2_train = r2_score(y_train,y_train_pred)
    mse_test = mean_squared_error(y_test,y_test_pred)
    mse_train = mean_squared_error(y_train,y_train_pred)


    print("#R2 Test: ",r2_test," MSE Test: ",mse_test)
    print("#R2 Train: ",r2_train," MSE Train: ",mse_train)

    plt.figure()
    plt.title("Dados de Test")
    plt.xlabel(feature_names[x_index])
    plt.ylabel(target_names[y_index])
    plt.scatter(X_test[:, x_index], y_test, label='Teste R2 Test: %.5f MSE Test: %.5f'%(r2_test,mse_test))
    plt.scatter(X_test[:, x_index], y_test_pred, label='Predito')
    plt.legend(loc=2,bbox_to_anchor=(0, 1.1))


    plt.figure()
    plt.title("Dados de Treino")
    plt.xlabel(feature_names[x_index])
    plt.ylabel(target_names[y_index])
    plt.scatter(X_train[:, x_index], y_train, label='Treino R2: %.5f MSE: %.5f'%(r2_train,mse_train))
    plt.scatter(X_train[:, x_index], y_train_pred, label='Predito')
    plt.legend(loc=2,bbox_to_anchor=(0, 1.1))



def plotXY(x,y,labelx,labely,legend,title,midle_line=False):
    plt.figure()
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.scatter(x,y, label=legend)
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    if midle_line:
        plt.plot([-1,2],[-1,2],'r')
        
    plt.legend(loc=2)


def pltCorrelation(model,feature_names):
    feature_importance = model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure()
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos,feature_names[sorted_idx])
    plt.xlabel('Importância Relativa')
    plt.title("Corelação das características")
   

def pltLossGraph(train,validation,legend_1="Treino",legend_2="Validação",title="Interações X Erro",label_x="Interações",label_y="Erro",step=1):

    plt.figure()
    plt.title(title)
    plt.plot((np.arange(len(train)))*step, train, 'b-',
            label=legend_1)
    plt.plot((np.arange(len(validation)))*step, validation, 'r-',
            label=legend_2)
    
    plt.legend(loc='upper right')
    plt.xlim(0)
    plt.ylim(0)
    
    plt.xlabel(label_x)
    plt.ylabel(label_y)
   
def pltShow():
    plt.show()

def plotLeanrningCurve(X_train,X_val,y_train,y_val,model,param_key,scoring='r2',label_x=None,label_y=None,legend_1=None,legend_2=None,title=None,verbose=0,step=1):
    
    param = model.get_params()
    param_max = param[param_key]
    train_score = []
    validation_score = []
    
    for i in range(1,param_max+step,step):
        par = {param_key:i}
        model.set_params(**par)
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_validation_pred = model.predict(X_val)
        score_train = 0
        score_validation = 0
        if scoring == 'r2':
            score_train = r2_score(y_train,y_train_pred)
            score_validation = r2_score(y_val,y_validation_pred)
            train_score.append(score_train)
            validation_score.append(score_validation)
        elif scoring == 'mean_squared_error':
            score_train = mean_squared_error(y_train,y_train_pred)
            score_validation = mean_squared_error(y_val,y_validation_pred)
            train_score.append(score_train)
            validation_score.append(score_validation)
    
        elif scoring == 'mean_absolute_error':
            score_train = mean_absolute_error(y_train,y_train_pred)
            score_validation = mean_absolute_error(y_val,y_validation_pred)
            train_score.append(score_train)
            validation_score.append(score_validation)
        else:
            print("Error in plotLeanrningCurve")
            print("Param Scoring invalid, this function acept only: 'r2','mean_squared_erro' , 'mean_absolute_error'")
            return
        if verbose:
            print("Iteração: %d / %d Score: %s Train Score: %.5f Validation Score: %.5f"%(i,param_max+1,scoring,score_train,score_validation))
    if scoring == 'r2':
        score_name = "R-squared"
    
    elif scoring == 'mean_squared_error':
        score_name = "Mean Squared Error" 
    elif scoring == 'mean_absolute_error':
        score_name = "Mean Absolute Error" 

    if not title:
        title = "%s X %s"%(param_key,score_name)
    if not label_x:
        label_x = param_key
    if not label_y:
        label_y = score_name
    par = {param_key:param_max}
    model.set_params(**par)
    pltLossGraph(train_score,validation_score,title=title,label_x=param_key,label_y=score_name,legend_1=legend_1,legend_2=legend_2,step=step)


def getMetrics(y_true,y_pred,verbose=0):
    r2 = r2_score(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    if verbose:
        print("#R-squared: %.5f"%(r2))
        print("#Mean Squared Error: %.5f"%(mse))
    return r2,mse


def getBalancedDataSetIndexRandomState(X,y,model,scoring='r2'):
    index1 = 0
    index2 = 0
    menor_erro = 1000000
    maior_r2 = 0    
    for k in range(1,10):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=k)
        for i in range(1,10):
            X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2,random_state=i) 
            model.fit(X_train,y_train)
            #y_train_pred = model.predict(X_train)
            #r2,mse = getMetrics(y_train,y_train_pred)
            
            y_val_pred = model.predict(X_val)
            r2,mse = getMetrics(y_val,y_val_pred)
            
            print(k,i , mse,r2)
            if scoring == 'mean_squared_error':
                if mse < menor_erro: 
                    menor_erro = mse
                    index2 = i
                    index1 = k
                    maior_r2 = r2
            elif scoring == 'r2':
                if r2 > maior_r2: 
                    menor_erro = mse
                    index2 = i
                    index1 = k
                    maior_r2 = r2
    
    print("random_state1:",index1,"random_state2:",index2,"R2",maior_r2,"MSE:", menor_erro)
    
    if scoring == 'r2':
        return index1,index2,r2
    elif scoring == 'mean_squared_error':
        return index1,index2,menor_erro
