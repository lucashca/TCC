from sklearn.metrics import mean_squared_error, r2_score,max_error
import matplotlib.pyplot as plt
from models.classes import Results


def processRegressor(regressor,dataSplited):
    maiorScore = -1000
    menorErroMedio = 1000
    menorErroMaximo = 1000    
    xTrain = dataSplited[0]
    xTest = dataSplited[1]
    yTrain = dataSplited[2]
    yTest = dataSplited[3]
    bestModel = [[regressor[0],maiorScore],[regressor[0],menorErroMedio],[regressor[0],menorErroMaximo]]
    total = len(regressor)
    for i,reg in enumerate(regressor):
        try:
            model = reg[1]
            model.fit(xTrain,yTrain)
            yPred = model.predict(xTest)
            atualScore = r2_score(yTest,yPred)
            atualErrorMedio = mean_squared_error(yTest,yPred)
            atualMaxError = max_error(yTest,yPred)
            print("%d/%d - Score: %.2f MSE: %.2f MXE:%.2f"%(i,total,maiorScore,menorErroMedio,menorErroMaximo))
            if atualScore > maiorScore:
                maiorScore = atualScore
                bestModel[0] = [reg,maiorScore]
            if atualErrorMedio < menorErroMedio:
                menorErroMedio = atualErrorMedio
                bestModel[1] = [reg,menorErroMedio]
            if atualMaxError < menorErroMaximo:
                menorErroMaximo = atualMaxError
                bestModel[2] = [reg,menorErroMaximo]

        except Exception as e:
            print(e)
    
    return bestModel

def processRegressorByMSE(regressor,dataSplited):
    menorErroMedio = 1000
    xTrain = dataSplited[0]
    xTest = dataSplited[1]
    yTrain = dataSplited[2]
    yTest = dataSplited[3]
    bestModel = Results()
    total = len(regressor)
    for i,reg in enumerate(regressor):
        try:
            model = reg[1]
            model.fit(xTrain,yTrain)
            yPred = model.predict(xTest)
            atualScore = r2_score(yTest,yPred)
            atualErrorMedio = mean_squared_error(yTest,yPred)
            atualMaxError = max_error(yTest,yPred)
            print("%d/%d - Score: %.2f MSE: %.2f MXE:%.2f"%(i,total,atualScore,menorErroMedio,atualMaxError))
            if atualErrorMedio < menorErroMedio:
                menorErroMedio = atualErrorMedio
                bestModel.regressor = reg
                bestModel.score = menorErroMedio
                
            

        except Exception as e:
            print(e)
    
    return bestModel




def processRegressorByR2(regressor,dataSplited):
    maiorScore = -1000  
    xTrain = dataSplited[0]
    xTest = dataSplited[1]
    yTrain = dataSplited[2]
    yTest = dataSplited[3]
    bestModel = Results(None,None)
    total = len(regressor)
    for i,reg in enumerate(regressor):
        try:
            model = reg[1]
            model.fit(xTrain,yTrain)
            yPred = model.predict(xTest)
            atualScore = r2_score(yTest,yPred)
            atualErrorMedio = mean_squared_error(yTest,yPred)
            atualMaxError = max_error(yTest,yPred)    
            if atualScore > maiorScore:
                maiorScore = atualScore
                bestModel.regressor = reg
                bestModel.score = atualScore
        
            if (i% int(total/10) == 0):  
               print("%d/%d - Score: %.2f MSE: %.2f MXE:%.2f"%(i,total,maiorScore,atualErrorMedio,atualMaxError))


        except Exception as e:
            print(e)


    
    return bestModel


def processRegressorMXE(regressor,dataSplited):
    menorErroMaximo = 1000    
    xTrain = dataSplited[0]
    xTest = dataSplited[1]
    yTrain = dataSplited[2]
    yTest = dataSplited[3]
    bestModel = Results(None,None)
    total = len(regressor)
    for i,reg in enumerate(regressor):
        try:
            model = reg[1]
            model.fit(xTrain,yTrain)
            yPred = model.predict(xTest)
            atualScore = r2_score(yTest,yPred)
            atualErrorMedio = mean_squared_error(yTest,yPred)
            atualMaxError = max_error(yTest,yPred)
            #print("%d/%d - Score: %.2f MSE: %.2f MXE:%.2f"%(i,total,atualScore,atualErrorMedio,menorErroMaximo))
            if atualMaxError < menorErroMaximo:
                menorErroMaximo = atualMaxError
                bestModel.regressor = reg
                bestModel.score = menorErroMaximo
                


        except Exception as e:
            print(e)
    
    return bestModel


def printChart(model,dataSplited,dataSetNormalizada,yColunm,regressorConfig):
    xTrain = dataSplited[0]
    xTest = dataSplited[1]
    yTrain = dataSplited[2]
    yTest = dataSplited[3]

    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)

    atualScore = r2_score(yTest,yPred)
    atualErrorMedio = mean_squared_error(yTest,yPred)
    atualMaxError = max_error(yTest,yPred)
    

    plt.suptitle("Score: %.5f MSE: %.5f MXE: %.5f \n %s "% (atualScore,atualErrorMedio,atualMaxError,regressorConfig), fontsize=16)   
    ax = plt.subplot(321,projection='3d')
    ax.scatter(dataSetNormalizada[:,0],dataSetNormalizada[:,1],dataSetNormalizada[:,yColunm] ,c=dataSetNormalizada[:,yColunm] ,cmap="coolwarm")
    ax.set_title("Full Data Set")
    ax.set_zlim3d(0, 1)

    ax1 = plt.subplot(322,projection='3d')
    ax1.scatter(xTrain[:,0], xTrain[:,1],yTrain,c=yTrain ,cmap="coolwarm")
    ax1.set_title("Train Data")
    ax1.set_zlim3d(0, 1)

    ax2 = plt.subplot(323,projection='3d')
    ax2.scatter(xTest[:,0], xTest[:,1],yTest,c=yTest ,cmap="coolwarm")
    ax2.set_title("Test Data")
    ax2.set_zlim3d(0, 1)

    ax3 = plt.subplot(324,projection='3d')
    ax3.scatter(xTest[:,0], xTest[:,1],yPred,c=yPred ,cmap="coolwarm")
    ax3.set_title("Pred Data")
    ax3.set_zlim3d(0, 1)

    ax3 = plt.subplot(325)
    ax3.scatter(xTest[:,0],yTest,c=yTest ,cmap="coolwarm")
    ax3.set_title("Test Data")
    plt.ylim(0, 1)
    

    ax3 = plt.subplot(326)
    ax3.scatter(xTest[:,0],yPred,c=yPred ,cmap="coolwarm")
    ax3.set_title("Pred Data")
    plt.ylim(0, 1)
    
    plt.show()