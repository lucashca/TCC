
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

import matplotlib.pyplot as plt


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


def findBalancedDataSet(faixa,X,y,searchMethod,negRefit=False):
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
        best_model,best_params,best_score =  searchMethod(X_train,y_train)
        
        if best_score*operador > global_best_score:
            global_best_score = best_score
            global_best_seed = r
            global_best_params = best_params
            global_best_model = best_model
        cont+=1


        print("%d / %d : Best Score: %.5f "%(cont,total,global_best_score))
    print()
    print("Conluido - Best Score: %.5f Semente: %d "%(global_best_score,global_best_seed))
    return global_best_model,global_best_params,global_best_score,global_best_seed



def pltResults(best_model,x_index,X_train,X_test,y_train,y_test):
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    r2_test = r2_score(y_test,y_test_pred)
    r2_train = r2_score(y_train,y_train_pred)
    mse_test = mean_squared_error(y_test,y_test_pred)
    mse_train = mean_squared_error(y_train,y_train_pred)


    print("R2 Test: ",r2_test," MSE Test: ",mse_test)
    print("R2 Train: ",r2_train," MSE Train: ",mse_train)

    plt.figure()
    plt.title("Dados de Test")
    plt.xlabel('Latitude')
    plt.ylabel('Ca')
    plt.scatter(X_test[:, x_index], y_test, label='Teste R2 Test: %.5f MSE Test: %.5f'%(r2_test,mse_test))
    plt.scatter(X_test[:, x_index], y_test_pred, label='Predito')
    plt.legend(loc=2,bbox_to_anchor=(0, 1.1))


    plt.figure()
    plt.title("Dados de Treino")
    plt.xlabel('Latitude')
    plt.ylabel('Ca')
    plt.scatter(X_train[:, x_index], y_train, label='Treino R2: %.5f MSE: %.5f'%(r2_train,mse_train))
    plt.scatter(X_train[:, x_index], y_train_pred, label='Predito')
    plt.legend(loc=2,bbox_to_anchor=(0, 1.1))

    plt.show()
