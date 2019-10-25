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

def deNormalizeColumn(dataSetOriginal,column,normalizedColum):
    original = []
    minimo = getMin(dataSetOriginal,column)
    maximo = getMax(dataSetOriginal,column)
    for i in range(len(normalizedColum)):
        n = normalizedColum[i]*(maximo - minimo)+minimo
        original.append(n)
    return original


def convertCSV(arr):
    #Corrige o dataset, convertendo strig para float e também coverte a unidade dos nutriente de mg/l para cmolc/dm3
    for data in arr:
        data[0] = float(data[0])
        data[1] = float(data[1])
        data[2] = float(data[2])/200.4#Ca
        data[3] = float(data[3])/121.56 #Mg
        data[4] = float(data[4])/230#Na
        data[5] = float(data[5])/391 #K
        data[6] = float(data[6])/506.47 #Cl
        
    return arr
