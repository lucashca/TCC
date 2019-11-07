
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, "../myTools")
from dataSetPreProcessing import train_validation_test_split
from loadDataSet import loadMainDataSet,loadTesteDataSet,loadCompletDataSet,loadMainDataSetWithElevation
from tools import verifyArgs,findBalancedDataSet,pltResults
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV







def GridSearchCVGradientBoostingRegressor(X_train,y_train):
    d = [i for i in range(1,100)]
    d.append(None)
    d = [10,20,30,40,50,60,70,80,90,100]
    e = [10,20,30,40,50,60,70,80,90,100]
    l = [2,4,5,11]
    d = [10,50,80]
    e = [10,50,80]
    
  
    param_grid={
        'n_estimators':e,#[10,50,80], 
        'learning_rate': [0.05,0.01], 
        'max_depth':d,#[10,50], 
        'min_samples_leaf':l, 
        'max_features':['auto','log2'],
        'random_state':range(1) 
    } 

    model = GradientBoostingRegressor(random_state=0,verbose=0)

    scores = [
        'r2',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
    ]
    reg =GridSearchCV(model, cv=10,param_grid=param_grid,verbose=0,n_jobs=-1,scoring=scores,refit='r2',iid=True)

    reg.fit(X_train,y_train)

    return reg.best_estimator_,reg.best_params_,reg.best_score_



args = sys.argv[1:]

y_column,random_state = verifyArgs(args)

# Load data set
dataSet = loadMainDataSetWithElevation()
#dataSet = loadTesteDataSet()
#dataSet = loadMainDataSet()
#dataSet = loadCompletDataSet()
#Set features and target

y_column = 2 
X = dataSet[:,0:4]
y = dataSet[:,5]


best_model , best_params, best_score,best_seed = findBalancedDataSet(range(1,10),X,y,GridSearchCVGradientBoostingRegressor)

print("#Best Params",best_params)
print("#Best Score",best_score)
print("#Best Seed",best_seed)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=best_seed)
pltResults(best_model,X.shape[1]-1,X_train,X_test,y_train,y_test)




## Modelo para o sódio
## Entrada latitude, longitude, elevação, cálcio
def MELHOR_RESULTADO_MG():
    X = dataSet[:,0:4]
    y = dataSet[:,4]
    param = {'learning_rate': [0.05], 'max_depth': [50], 'max_features': ['log2'], 'min_samples_leaf': [11], 'n_estimators': [60]}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=9)
    model =GradientBoostingRegressor(random_state=0)
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print("Cross Validation  R2 Score :",grid.best_score_)
    print(grid.best_params_)
    pltResults(best_model,1,X_train,X_test,y_train,y_test)
    #Conluido - Best Score: 0.73465 Semente: 9 
    #Best Params {'learning_rate': 0.05, 'max_depth': 50, 'max_features': 'log2', 'min_samples_leaf': 11, 'n_estimators': 60, 'random_state': 0}
    #Best Score 0.7346493872397788
    #Best Seed 9
    #R2 Test:  0.7641468991808511  MSE Test:  0.0037531633086605658
    #R2 Train:  0.8748115698686528  MSE Train:  0.003076693813010081

    MELHOR_RESULTADO_MG()

def MELHOR_RESULTADO_NA():
    X = dataSet[:,0:4]
    y = dataSet[:,5]
    param = {'learning_rate': [0.05], 'max_depth': [20], 'max_features': ['log2'], 'min_samples_leaf': [5], 'n_estimators': [50], }
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=1)
    model =GradientBoostingRegressor(random_state=0)
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print("Cross Validation  R2 Score :",grid.best_score_)
    
    pltResults(best_model,1,X_train,X_test,y_train,y_test)
    
    #Conluido - Best Score: 0.73448 Semente: 9 
    #Best Params {'learning_rate': 0.05, 'max_depth': 10, 'max_features': 2, 'min_samples_leaf': 5, 'n_estimators': 50, 'random_state': 0}
    #Best Score 0.734480868500437
    MELHOR_RESULTADO_NA()

def MELHOR_RESULTADO_K():
    X = dataSet[:,0:4]
    y = dataSet[:,6]
    param = {'learning_rate': [0.05], 'max_depth': [30], 'max_features': ['log2'], 'min_samples_leaf': [4], 'n_estimators': [90], }
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=3)
    model =GradientBoostingRegressor(random_state=0)
    grid = GridSearchCV(model,param, cv=10,verbose=0,n_jobs=-1,scoring='r2',iid=True)
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_ 
    print(grid.best_score_)
    print(grid.best_params_)
    pltResults(best_model,0,X_train,X_test,y_train,y_test)
    #Conluido - Best Score: 0.63825 Semente: 3 
    #Best Params {'learning_rate': 0.05, 'max_depth': 30, 'max_features': 'log2', 'min_samples_leaf': 4, 'n_estimators': 90, 'random_state': 0}
    #Best Score 0.6382454854124988
    #R2 Test:  0.6772838242726009  MSE Test:  0.0078083038481006
    #R2 Train:  0.9892885578561049  MSE Train:  0.00025464662511601085
    
    MELHOR_RESULTADO_K()


############### MG ##########################
#Conluido - Best Score: 0.65662 Semente: 9 
#Best Params {'learning_rate': 0.05, 'max_depth': 7, 'max_features': 'auto', 'min_samples_leaf': 5, 'n_estimators': 19, 'random_state': 0}
#Best Score 0.6566162097362611
#Best Seed 9
#R2 Test:  0.6698401069997604  MSE Test:  0.005253880453960908
#R2 Train:  0.7738415101819238  MSE Train:  0.0055581847751659365


#Conluido - Best Score: 0.73203 Semente: 9 
#Best Params {'learning_rate': 0.05, 'max_depth': 50, 'max_features': 'log2', 'min_samples_leaf': 5, 'n_estimators': 50, 'random_state': 0}
#Best Score 0.7320304172385002
#Best Seed 9
#R2 Test:  0.6928402471363506  MSE Test:  0.004887876014100292
#R2 Train:  0.9412048892640388  MSE Train:  0.0014449782080243406


#Conluido - Best Score: 0.73465 Semente: 9 
#Best Params {'learning_rate': 0.05, 'max_depth': 50, 'max_features': 'log2', 'min_samples_leaf': 11, 'n_estimators': 60, 'random_state': 0}
#Best Score 0.7346493872397788
#Best Seed 9
#R2 Test:  0.7641468991808511  MSE Test:  0.0037531633086605658
#R2 Train:  0.8748115698686528  MSE Train:  0.003076693813010081


############### Na ########################## 


#Conluido - Best Score: 0.59157 Semente: 2 
#Best Params {'learning_rate': 0.05, 'max_depth': 50, 'max_features': 'log2', 'min_samples_leaf': 5, 'n_estimators': 50, 'random_state': 0}
#Best Score 0.5915734707430875
#Best Seed 2
#R2 Test:  0.6002369455399681  MSE Test:  0.012816895257928434
#R2 Train:  0.9122383051241635  MSE Train:  0.0019139624633444869


#Conluido - Best Score: 0.59157 Semente: 2 
#Best Params {'learning_rate': 0.05, 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 5, 'n_estimators': 50, 'random_state': 0}
#Best Score 0.5915734707430875
#Best Seed 2
#R2 Test:  0.6002369455399681  MSE Test:  0.012816895257928434
#R2 Train:  0.9122383051241635  MSE Train:  0.0019139624633444869

#Conluido - Best Score: 0.59717 Semente: 2 
#Best Params {'learning_rate': 0.05, 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 5, 'n_estimators': 50, 'random_state': 0}
#Best Score 0.5971743288694583
#Best Seed 2
#R2 Test:  0.6153244736541088  MSE Test:  0.012333170548047006
#R2 Train:  0.8709102998267071  MSE Train:  0.002815269701498251


############### K #############################

#Conluido - Best Score: 0.63825 Semente: 3 
#Best Params {'learning_rate': 0.05, 'max_depth': 30, 'max_features': 'log2', 'min_samples_leaf': 4, 'n_estimators': 90, 'random_state': 0}
#Best Score 0.6382454854124988
#Best Seed 3
#R2 Test:  0.6772838242726009  MSE Test:  0.0078083038481006
#R2 Train:  0.9892885578561049  MSE Train:  0.00025464662511601085