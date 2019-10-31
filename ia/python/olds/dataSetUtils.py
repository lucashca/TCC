
from sklearn.model_selection import train_test_split




def convertCSV(arr):
    #Corrige o dataset, convertendo strig para float e também coverte a unidade dos nutriente de mg/l para cmolc/dm3
    for data in arr:
        data[0] = float(data[0])
        data[1] = float(data[1])
        data[2] = round(float(data[2])/200.4,2) #Ca
        data[3] = round(float(data[3])/121.56,2) #Mg
        data[4] = round(float(data[4])/230,2) #Na
        data[5] = round(float(data[5])/391,2) #K
        data[6] = round(float(data[6])/506.47,2) #Cl
        
    return arr

def getMin(dataSet,column):
    minimo = dataSet[0][column]
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        if(d < minimo):
            minimo = d
    return minimo    

def getMax(dataSet,column):
    maximo = dataSet[0][column]
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        if(d > maximo):
            maximo = d
    return maximo    

def normalizeColumn(dataSet,column):
    minimo = getMin(dataSet,column)
    maximo = getMax(dataSet,column)

    for i in range(len(dataSet)):
        d = dataSet[i][column]
        dataSet[i][column] = (d - minimo)/(maximo - minimo)



def saveDump(data,name):
    fname = "./dump/"+str(name)+".csv"
    csv_file = open(fname, "w")
    c = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    c.writerow(["Config","R2-Score","MSE","Max Error"])
    c.writerow(data)

def splitDataForTrain(dataSet,yColunm,random_state = 0,test_size=0.3):
    xTrain,xTest,yTrain,yTest = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=test_size,random_state=random_state)
    return [xTrain,xTest,yTrain,yTest]