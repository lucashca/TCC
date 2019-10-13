from sklearn.ensemble import RandomForestRegressor

def createRandomForrestRegressor():
    regressor = []
    for depth in range(10):
        for r in range(10):
            #regression.append([[depth,r],DecisionTreeRegressor(max_depth=depth+1,random_state=r)])   
            for s in range(10):
                regressor.append([["RandomForestRegressor",depth,r,s],RandomForestRegressor(max_depth=depth+1, random_state=r,n_estimators=s+1)])
    print("Regressor created!")
    return regressor
