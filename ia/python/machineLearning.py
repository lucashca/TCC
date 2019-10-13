from sklearn.metrics import mean_squared_error, r2_score,max_error
import matplotlib.pyplot as plt

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
    bestModel = []
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
            if atualErrorMedio < menorErroMedio:
                menorErroMedio = atualErrorMedio
                bestModel = reg
            

        except Exception as e:
            print(e)
    
    return bestModel




def processRegressorByR2(regressor,dataSplited):
    maiorScore = -1000  
    xTrain = dataSplited[0]
    xTest = dataSplited[1]
    yTrain = dataSplited[2]
    yTest = dataSplited[3]
    bestModel = {}
    total = len(regressor)
    for i,reg in enumerate(regressor):
        try:
            model = reg[1]
            model.fit(xTrain,yTrain)
            yPred = model.predict(xTest)
            atualScore = r2_score(yTest,yPred)
            atualErrorMedio = mean_squared_error(yTest,yPred)
            atualMaxError = max_error(yTest,yPred)
            print("%d/%d - Score: %.2f"%(i,total,maiorScore))
            if atualScore > maiorScore:
                maiorScore = atualScore
                bestModel.set("regression",reg)
                bestModel.set("score",maiorScore)
               
            
            

        except Exception as e:
            print(e)
    
    return bestModel


def processRegressorMXE(regressor,dataSplited):
    menorErroMaximo = 1000    
    xTrain = dataSplited[0]
    xTest = dataSplited[1]
    yTrain = dataSplited[2]
    yTest = dataSplited[3]
    bestModel = []
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
            if atualMaxError < menorErroMaximo:
                menorErroMaximo = atualMaxError
                bestModel = reg


        except Exception as e:
            print(e)
    
    return bestModel


def printChart(model,dataSplited,dataSetNormalizada,yColunm):
    xTrain = dataSplited[0]
    xTest = dataSplited[1]
    yTrain = dataSplited[2]
    yTest = dataSplited[3]

    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)

    atualScore = r2_score(yTest,yPred)
    atualErrorMedio = mean_squared_error(yTest,yPred)
    atualMaxError = max_error(yTest,yPred)
    
        
    ax = plt.subplot(321,projection='3d')
    ax.scatter(dataSetNormalizada[:,0],dataSetNormalizada[:,1],dataSetNormalizada[:,yColunm] ,c=dataSetNormalizada[:,yColunm] ,cmap="coolwarm")
    ax.set_title("Score: %.5f MSE: %.5f MXE: %.5f "% (atualScore,atualErrorMedio,atualMaxError))
    
    ax1 = plt.subplot(322,projection='3d')
    ax1.scatter(xTrain[:,0], xTrain[:,1],yTrain,c=yTrain ,cmap="coolwarm")
    ax1.set_title("Train Data")

    ax2 = plt.subplot(323,projection='3d')
    ax2.scatter(xTest[:,0], xTest[:,1],yTest,c=yTest ,cmap="coolwarm")
    ax2.set_title("Test Data")


    ax3 = plt.subplot(324,projection='3d')
    ax3.scatter(xTest[:,0], xTest[:,1],yPred,c=yPred ,cmap="coolwarm")
    ax3.set_title("Pred Data")

    ax3 = plt.subplot(325)
    ax3.scatter(xTest[:,0],yTest,c=yTest ,cmap="coolwarm")
    ax3.set_title("Pred Data")

    ax3 = plt.subplot(326)
    ax3.scatter(xTest[:,0],yPred,c=yPred ,cmap="coolwarm")
    ax3.set_title("Pred Data")

    plt.show()