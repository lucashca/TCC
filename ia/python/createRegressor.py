from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

def createRandomForrestRegressor():
    regressor = []
    for depth in range(10):
        for r in range(10):
            #regression.append([[depth,r],DecisionTreeRegressor(max_depth=depth+1,random_state=r)])   
            for s in range(10):
            #s = 10
                if depth == 0:
                    regressor.append([["RandomForestRegressor",None,r,s+1],RandomForestRegressor(criterion="mse",max_depth=None, random_state=r,n_estimators=s+1)])
                regressor.append([["RandomForestRegressor",depth+1,r,s+1],RandomForestRegressor(criterion="mse",max_depth=depth+1, random_state=r,n_estimators=s+1)])
        
    print("Regressor created!")
    return regressor

def createDecisionTreeRegressor():
    splitter = ["best","random"]

    regressor = []
    for depth in range(10):
        for r in range(1):
            #regression.append([[depth,r],DecisionTreeRegressor(max_depth=depth+1,random_state=r)])   
            for s in splitter:
            #s = 10
                if depth == 0:
                    regressor.append([["DecisionTreeRegressor",None,r,s],DecisionTreeRegressor(criterion="mse",max_depth=None, random_state=r,splitter=s)])
                regressor.append([["DecisionTreeRegressor",depth+1,r,s],DecisionTreeRegressor(criterion="mse",max_depth=depth+1, random_state=r,splitter=s)])
        
    print("Regressor created!")
    return regressor

def createMLPRegressor():
    ACTIVATION_TYPES = ["relu","logistic","identity"]
    SOLVER_TYPES = ["lbfgs"]
    ALPHA = [0.00001,0.001,0.01,0.1,1]
    LEARNING_RATE_TYPES = ["constant","invscaling","adaptative"]

    regression = []
    for activation in ACTIVATION_TYPES:
        for solver in SOLVER_TYPES:
            for alfa in ALPHA:
                for lrt in LEARNING_RATE_TYPES:
                    for r in range(10):
                        if lrt == "adaptative" and activation == "sgd":
                            regression.append([[activation,solver,alfa,lrt,r],MLPRegressor(hidden_layer_sizes=(50,50,50),activation=activation,solver=solver,alpha=alfa,learning_rate=lrt,max_iter=1000,random_state=r)])
                        elif not lrt == "adaptative" :
                            regression.append([[activation,solver,alfa,lrt,r],MLPRegressor(hidden_layer_sizes=(20,20),activation=activation,solver=solver,alpha=alfa,learning_rate=lrt,max_iter=1000,random_state=r)])
    return regression