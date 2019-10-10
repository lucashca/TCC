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

csv = pd.read_csv("dataSetCompleto.csv",usecols=[0,1,2,3,4,5,6])


def fixCsv(arr):
    #Corrige o dataset, convertendo strig para float e também coverte a unidade dos nutriente de mg/l para cmolc/dm3
    for data in arr:
        data[0] = float(data[0].replace(',','.'))
        data[1] = float(data[1].replace(',','.'))
        data[2] = round(float(data[2])/200.4,2) #Ca
        data[3] = round(float(data[3])/121.56,2) #Mg
        data[4] = round(float(data[4])/230,2) #Na
        data[5] = round(float(data[5])/391,2) #K
        data[6] = round(float(data[6])/506.47,2) #Cl
        
    return arr


def getMaxValues(data):
    maxValues = []
    for i in range(data.shape[1]):
        maxValues.append(max(abs(data[:,i])))
    return maxValues

def getMinValues(data):
    minValues = []
    for i in range(data.shape[1]):
        minValues.append(min(abs(data[:,i])))
    return minValues

def getMedias(data):
    medias = []
    for i in range(data.shape[1]):
        medias.append(sum(data[:,i])/data.shape[0])
    return medias

def getMaiorMedia(data,media):
    maiorMedia = []
    for i in range(data.shape[1]):
        m = []
        for d in data[:,i]:
            if(d > media[i]):
                m.append(d)
        maiorMedia.append(m)
    return maiorMedia

def getMenorMedia(data,media):
    maiorMedia = []
    for i in range(data.shape[1]):
        m = []
        for d in data[:,i]:
            if(d < media[i]):
                m.append(d)
        maiorMedia.append(m)
    return maiorMedia

def normalizeData(dataSet):
    maximos = getMaxValues(dataSet)
    minimos = getMinValues(dataSet)
    data = np.zeros(dataSet.shape)
    
    for i in range(dataSet.shape[1]):
        data[:,i] = (abs(dataSet[:,i]) - minimos[i])/(maximos[i] - minimos[i])
    return data





dataSet = np.array(fixCsv(csv.values))
maxValues = getMaxValues(dataSet)
medias = getMedias(dataSet)
maiorMedia = getMaiorMedia(dataSet,medias)
menorMedia = getMenorMedia(dataSet,medias)


data = normalizeData(dataSet)

dataFrameOriginal = pd.DataFrame({"Latitude":dataSet[:,0],"Longitude":dataSet[:,1],"Ca":dataSet[:,2],"Mg":dataSet[:,3],"Na":dataSet[:,4],"K":dataSet[:,5],"Cl":dataSet[:,6]})
dataFrameNormalizado = pd.DataFrame({"Latitude":data[:,0],"Longitude":data[:,1],"Ca":data[:,2],"Mg":data[:,3],"Na":data[:,4],"K":data[:,5],"Cl":data[:,6]})

dataColun = 3
xTrain, xTest, yTrain, yTest = train_test_split(data[:,0:2], data[:,dataColun], test_size = 0.4, random_state = 0)


regression = [
    ["SVR - RBF", SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)],
    ["SVR - Linear",SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)],
    ["SVR - Poly",SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)],
    ["Linear Model",linear_model.LinearRegression()  ],
    ["Random Forrest Regressor",RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)],
    ["Decision Tree Regression 1", DecisionTreeRegressor(max_depth=1)],
    ["Mult Layer Perceptron",MLPRegressor(solver='lbfgs', learning_rate ="adaptive",alpha=1,hidden_layer_sizes=(100), random_state=1)],
    ["Ridge",linear_model.Ridge(alpha=.5)],
    ["RidgeCV",linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))],
    ["Lasso",linear_model.Lasso(alpha=0.1)],
    ["ElasticNet",linear_model.ElasticNet(alpha=0.1)],
    ["LassoLars",linear_model.LassoLars(alpha=0.1)],
    ["OrthogonalMatchingPursuit",linear_model.OrthogonalMatchingPursuit()],
    #["Logistic Regression", linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')]
    ["SGDRegressor",linear_model.SGDRegressor()],
    ["BayesianRidge",linear_model.BayesianRidge()],
    ["ARDRegression",linear_model.ARDRegression()],
    ["PassiveAggressiveRegressor",linear_model.PassiveAggressiveRegressor()],
    ["TheilSenRegressor",linear_model.TheilSenRegressor()],
     
  

]

for i in range(100):
    regression.append(["Decision Tree Regression %.1f"%i, DecisionTreeRegressor(max_depth=i+1)]),
    regression.append(["Random Forrest Regressor %.1f"%i, RandomForestRegressor(max_depth=i+1, random_state=0,n_estimators=100)])
    regression.append(["Mult Layer Perceptron",MLPRegressor(solver='lbfgs', learning_rate ="adaptive",alpha=1,hidden_layer_sizes=(i+1), random_state=1)],
    )

model = 0
maior = 0
maior2 = 0
bestYpred = []
regFunction = []
for i,reg in enumerate(regression):
    print(reg[0])
    reg[1].fit(xTest,yTest)
    yPred = reg[1].predict(xTest)
    score = reg[1].score(xTest,yTest)
    if score >= maior:
        score2 = reg[1].score(xTrain,yTrain)
        #if score2 >= maior2:
        regFunction = reg
        model = reg[0]   
        bestYpred = yPred
        maior2 = score2
        maior = score
        

    print(reg[0]+" Score: %.2f" % score)
    print(reg[0]+" R-Sqared: %.2f" % r2_score(yTest, yPred))
    print("Mean squared error: %.2f" % mean_squared_error(yTest, yPred))
    print("**************\n\n")


regression = regFunction[1]


regression.fit(xTrain,yTrain)
yPred = regression.predict(xTest)



print("Melhor: ",regFunction[0])
print("Score: ",maior)
print("Score2: ",maior2)

print("Mean squared error: %.2f" % mean_squared_error(yTest, yPred))
print("Explained Variance Score	 : %.2f" % metrics.explained_variance_score(yTest,yPred))    
print("Max Error	 : %.2f" % metrics.max_error(yTest,yPred))    
print("Mean Absolute Error	 : %.2f" % metrics.mean_absolute_error(yTest,yPred))    
print("Mean Squared Error	 : %.2f" % metrics.mean_squared_error(yTest,yPred))    
print("Mean Squared Log Error	 : %.2f" % metrics.mean_squared_log_error(yTest,yPred))    
print("Median Absolute Error	 : %.2f" % metrics.median_absolute_error(yTest,yPred))    
print("R2	 : %.2f" % metrics.r2_score(yTest,yPred))    


#print("Mean squared error: %.2f"% mean_squared_error(yTest, yPred))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(yTest, yPred))
# The coefficients
# The mean squared error

#for i,prediction in enumerate(yPred):
#    print("Predicted : %s, Target: %s" %(prediction,yTest[i]))

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot outputs

ax.scatter(xTrain[:,0],xTrain[:,1], yTrain,  color='black')

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Correto')

'''
fig2 = plt.figure()

ax2 = fig2.add_subplot(111,projection="3d")
ax2.scatter(xTest[:,0],xTest[:,1], yPred,c=yPred,cmap="coolwarm")
ax2.set_xlabel('Latitude')
ax2.set_ylabel('Longitude')
#ax2.set_zlabel('Predito')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111,projection="3d")

ax3.scatter(xTest[:,0],xTest[:,1], yTest,c=yTest,cmap="coolwarm")
ax3.set_xlabel('Latitude')
ax3.set_ylabel('Longitude')
#ax3.set_zlabel('Correto')


plt.show()
