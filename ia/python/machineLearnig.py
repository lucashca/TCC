
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,max_error
from multiprocessing import Process, Lock ,Manager


def printList(shared_list):
    for l in shared_list:
        print(l)

def splitArray(totalSize,workers):
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

def createProcessByBestRes(qtdWorker,regressor,dataSet,k,yColunm):
    totalSize = len(regressor)
    work = splitArray(totalSize,qtdWorker)
    shared_list = Manager().list([])
    process = []
    
    for i in range(qtdWorker):
       
        start = work[i][0]
        end = work[i][-1] + 1
        
        #t = Process(target=findBestModelByMedia,args=(regressor[start:end],dataSet,"Thread"+str(i),shared_list,k,yColunm))
        t = Process(target=findBestModelByBestRes,args=(regressor[start:end],dataSet,"Thread"+str(i),shared_list,k,yColunm))
        
        process.append(t)
        process[i].start()
    for i in range(qtdWorker):
        process[i].join()        


def createProcessByBestMean(qtdWorker,regressor,dataSet,k,yColunm):
    totalSize = len(regressor)
    work = splitArray(totalSize,qtdWorker)
    shared_list = Manager().list([])
    process = []
    
    for i in range(qtdWorker):
       
        start = work[i][0]
        end = work[i][-1] + 1
        
        t = Process(target=findBestModelByMedia,args=(regressor[start:end],dataSet,"Thread"+str(i),shared_list,k,yColunm))
        #t = Process(target=findBestModelByBestRes,args=(regressor[start:end],dataSet,"Thread"+str(i),shared_list,k,yColunm))
        
        process.append(t)
        process[i].start()
    for i in range(qtdWorker):
        process[i].join()     


def findBestModelByMedia(regressor,dataSet,tname,shared_list,k,yColunm):
  
    melhorMediaR2 = 0
    melhorMediaMSE = 0
    bestConfigR2 = []
    bestConfigMSE = []
    totalTestData = len(k)
    totalReg = len(regressor)
    cont = 0
    indexPercent  = 1
    for params,model in regressor:
        mediaR2 = 0
        mediaMSE = 0

        for i in k:
            x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=i)
            
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test,y_pred)
            mse = mean_squared_error(y_test,y_pred)
        
            mediaR2 += r2
            mediaMSE += mse
        
        mediaMSE = mediaMSE /totalTestData
        mediaR2 = mediaR2 /totalTestData

        if(cont == 0):
            melhorMediaMSE = mediaMSE
            melhorMediaR2 = mediaR2
            bestConfigMSE = params
            bestConfigR2 = params
            
        else:
            if(mediaMSE < melhorMediaMSE):
                melhorMediaMSE = mediaMSE
                bestConfigMSE = params
                
            if(mediaR2 > melhorMediaR2): 
                melhorMediaR2 = mediaR2
                bestConfigR2 = params

        cont +=1
        percent = cont*100/totalReg
        if percent > indexPercent*10:
            #print("%s : %d/%d - Melhor Media MSE: %.3f Melhor Media R2: %.3f"%(tname,cont,len(regressor),melhorMediaMSE,melhorMediaR2))
            print(percent)
            indexPercent+=1
    print("********* %s ***********"%(tname))
    print("Melhor média MSE %.5f"%(melhorMediaMSE))
    print("Melhor Configuração MSE ")
    print(bestConfigMSE)
    print("Melhor média R2 %.5f"%(melhorMediaR2))
    print("Melhor Configuração R2 ")
    print(bestConfigR2)
    print("@@@@@@@@@ %s @@@@@@@@@@"%(tname))
    shared_list.append([melhorMediaMSE,melhorMediaR2,bestConfigMSE,bestConfigR2])
    printList(shared_list)


def findBestModelByBestRes(regressor,dataSet,tname,shared_list,k,yColunm):
  
    melhorR2 = 0
    melhorMSE = 0
    bestConfigR2 = []
    bestConfigMSE = []
    faixa = k
    cont = 0
    totalReg = len(regressor)
    indexPercent = 1
    for params,model in regressor:
        
        for i in faixa:
          
            x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=i)
            
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test,y_pred)
            mse = mean_squared_error(y_test,y_pred)
    
            if(cont == 0):
                melhorMSE = mse
                melhorR2 = r2
                bestConfigMSE = []
                bestConfigR2 = []
                bestConfigMSE = [params[0],params[1],params[2],params[3],params[4],i]
                bestConfigR2 = [params[0],params[1],params[2],params[3],params[4],i]
         

            else:
                if(mse < melhorMSE):
                    melhorMSE = mse
                    bestConfigMSE = []
                    bestConfigMSE = [params[0],params[1],params[2],params[3],params[4],i]
                    
                    
                if(r2 > melhorR2): 
                    melhorR2 = r2
                    bestConfigR2 = []
                    bestConfigR2 = [params[0],params[1],params[2],params[3],params[4],i]
          
          
        cont +=1
        percent = cont*100/totalReg
   
        if percent > indexPercent*2:
            print("%s : %.2f %% %d/%d - MSE: %.3f R2: %.3f"%(tname,percent,cont,len(regressor),melhorMSE,melhorR2),bestConfigMSE,bestConfigR2)
            indexPercent+=1
    print("********* %s ***********"%(tname))
    print("Melhor MSE %.5f"%(melhorMSE))
    print("Melhor Configuração MSE ")
    print(bestConfigMSE)
    print("Melhor média R2 %.5f"%(melhorR2))
    print("Melhor Configuração R2 ")
    print(bestConfigR2)
    print("@@@@@@@@@ %s @@@@@@@@@@"%(tname))
    shared_list.append([melhorMSE,melhorR2,bestConfigMSE,bestConfigR2])
    printList(shared_list)