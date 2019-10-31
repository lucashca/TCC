
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,max_error
from multiprocessing import Process, Lock ,Manager
from models.classes import RegressorModel,RegressorResult
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

class GridData:
    
    regressionResults = []
    regressionModels = []
    dataSets = []

    def __init__(self,dataSet,dataVariation,models,workers,yColunm):
        self.dataSet = dataSet
        self.dataVariation = dataVariation
        self.regressionModels = models
        self.workers = workers
        self.yColunm = yColunm
        self.createDataSets()
        self.createWorkers(workers)

    def createDataSets(self):
        for i in range(self.dataVariation):
            xTrain,xTest,yTrain,yTest = train_test_split(self.dataSet[:,:2],self.dataSet[:,self.yColunm],test_size=0.3,random_state=i)
            self.dataSets.append([xTrain,xTest,yTrain,yTest])
   
    def findBestModel(self,regression,name,shared_list):
        regressorResult:RegressorResult = RegressorResult()
        maiorR2 = 0
        menorMse = 1000
        melhorMediaR2 = 0
        melhorMediaMSE = 0
        print(len(regression))
        cont = 0
        for params,model in regression:
            somaR2 = 0
            somaErroMSE = 0
            for data in self.dataSets:
                model.fit(data[0],data[2])
                yPred = model.predict(data[1])
                s = r2_score(data[3],yPred)
                somaR2 = somaR2 + s
                
                mse = mean_squared_error(data[3],yPred)
                somaErroMSE = somaErroMSE + mse
                
                if(s > maiorR2):
                    maiorR2 = s
                    regressorResult.bestR2 = RegressorModel(params,model,data,maiorR2)
                if(mse < menorMse):
                    menorMse = mse
                    regressorResult.bestMSE= RegressorModel(params,model,data,menorMse)
                
            mediaR2 = somaR2/len(self.dataSets)
            mediaMSE = somaErroMSE/len(self.dataSets)
            cont += 1
            print("%s: %d/%d"%(name,cont,len(regression)))

            if(mediaR2 > melhorMediaR2):
                melhorMediaR2 = mediaR2
                regressorResult.bestMeanR2 = RegressorModel(params,model,None,melhorMediaR2)


            if(mediaMSE > melhorMediaMSE ):
                melhorMediaMSE = mediaMSE
                regressorResult.bestMeanMSE = RegressorModel(params,model,None,melhorMediaMSE)


        print("******"+name+"******")
        print("Maior Media R2",regressorResult.bestMeanR2.score)
        print("Maior Media MSE",regressorResult.bestMeanMSE.score)
        print("Maior R2",regressorResult.bestR2.score)
        print("Menor MSE",regressorResult.bestMSE.score)
        shared_list.append(regressorResult)
        self.printBest(shared_list)
    
    def createWorkers(self,workers):
        totalSize = len(self.regressionModels)
        work = self.splitArray(totalSize,workers)
        shared_list = Manager().list([])
        process = []
        for i in range(workers):
            
            start = work[i][0]
            end = work[i][-1] + 1
           
            #print(work[i])
            t = Process(target=self.findBestModel,args=(self.regressionModels[start:end],"Thread"+str(i),shared_list))
            process.append(t)
            process[i].start()
        for i in range(workers):
            process[i].join()        

    
    def splitArray(self,totalSize,workers):
     
        work = []
        if(totalSize%workers == 0):
            carga = int(totalSize/workers)
        else:
            carga = int(totalSize/workers) + 1
        for i in range(workers):
            inicio = i*carga
            fim = i*carga + carga
            if fim > totalSize:
                fim = totalSize
            r = range(inicio,fim)
            work.append(r)
        
        return work

    def appendResult(self,regressorResult:RegressorResult):
        self.regressionResults.append(regressorResult)
        print("Adicionado com sucesso")

    def printBest(self,shared_list):
        print(len(shared_list))
        if(len(shared_list) == self.workers):
            
            rr = RegressorResult()
            maiorR2 = 0
            menorMSE = 1000
            maiorMeanR2 = 0
            maiorMeanMSE = 0
            for r in shared_list:
                if r.bestMeanMSE.score > maiorMeanMSE:
                    maiorMeanMSE = r.bestMeanMSE.score
                    rr.bestMeanMSE = r.bestMeanMSE
                if r.bestMeanR2.score > maiorMeanR2:
                    maiorMeanR2 = r.bestMeanR2.score
                    rr.bestMeanR2 = r.bestMeanR2
                if r.bestMSE.score < menorMSE:
                    menorMSE = r.bestMSE.score
                    rr.bestMSE = r.bestMSE
                if r.bestR2.score > maiorR2:
                    maiorR2 = r.bestR2.score
                    rr.bestR2 = r.bestR2
                    
                   
            print("Maior R2",rr.bestR2.score)
            print("Maior Media R2",rr.bestMeanR2.score)
            print("Menor MSE",rr.bestMSE.score)
            print("Maior Media MSe",rr.bestMeanMSE.score)
            

            self.printChart(rr.bestMeanR2,"MeanR2: %.4f"%(maiorMeanR2))
            self.printChart(rr.bestMeanMSE,"MeanMSE:%.4f"%(maiorMeanMSE))
            self.printChart(rr.bestR2,"R2:%.4f"%(maiorR2))
            self.printChart(rr.bestMSE,"MSE:%.4f"%(menorMSE))

    def printChart(self,result:RegressorModel,name):
        dataSplited = result.data
        if dataSplited == None:
            dataSplited = self.dataSets[0]
        xTrain = dataSplited[0]
        xTest = dataSplited[1]
        yTrain = dataSplited[2]
        yTest = dataSplited[3]

        model = result.model
        model.fit(xTrain,yTrain)
        yPred = model.predict(xTest)
        atualScore = r2_score(yTest,yPred)
        atualErrorMedio = mean_squared_error(yTest,yPred)
        atualMaxError = max_error(yTest,yPred)
        plt.suptitle("%s Score: %.5f MSE: %.5f MXE: %.5f \n %s "% (name,atualScore,atualErrorMedio,atualMaxError,result.params), fontsize=16)   
        ax = plt.subplot(321,projection='3d')
        ax.scatter(self.dataSet[:,0],self.dataSet[:,1],self.dataSet[:,self.yColunm] ,c=self.dataSet[:,self.yColunm] ,cmap="coolwarm")
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
