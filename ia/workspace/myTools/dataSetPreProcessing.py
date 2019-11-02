from sklearn.model_selection import train_test_split


def getMin(dataSet,column):
    '''This function return the min value in a column'''
    minimo = dataSet[0][column]
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        if(d < minimo):
            minimo = d
    return minimo    

def getMax(dataSet,column):
    '''This function return the max value in a column'''
    maximo = dataSet[0][column]
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        if(d > maximo):
            maximo = d
    return maximo    



def train_validation_test_split(X,y,test_size=0.2,validation_size=0.2,random_state=0):
    '''This function return the train,validation and test data'''
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=validation_size)
    return X_train,X_val,X_test,y_train,y_val,y_test




def normalizeVal(val,minimo,maximo):
    '''This function normalize a single value
    minimo is the minimum value of the column to which the input data belongs.
    maximo is the minimum value of the column to which the input data belongs.
    '''
    d = (val - minimo)/(maximo - minimo)
    return d

def denormalizeVal(val,minimo,maximo):
    '''This function denormalize a single value
    minimo is the minimum value of the column to which the input data belongs.
    maximo is the minimum value of the column to which the input data belongs.
    '''
    d = val*(maximo - minimo) + minimo
    return d
    
def normalizeColumn(dataSet,column):
    '''This function normalize a column in the dataset'''
    minimo = getMin(dataSet,column)
    maximo = getMax(dataSet,column)
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        dataSet[i][column] = (d - minimo)/(maximo - minimo)

def deNormalizeColumn(dataSetOriginal,column,normalizedColum):
    '''''This function denormalize a column in the dataset'''
    original = []
    minimo = getMin(dataSetOriginal,column)
    maximo = getMax(dataSetOriginal,column)
    for i in range(len(normalizedColum)):
        n = normalizedColum[i]*(maximo - minimo)+minimo
        original.append(n)
    return original


def convertCSV(arr):
    '''This function convert nutriente vals form mg/l to cmolc/dm3'''
    #Corrige o dataset, convertendo strig para float e também coverte a unidade dos nutriente de mg/l para cmolc/dm3
    for data in arr:
        data[0] = float(data[0])
        data[1] = float(data[1])
        data[2] = round(float(data[2])/200.4,2)#Ca
        data[3] = round(float(data[3])/121.56,2) #Mg
        data[4] = round(float(data[4])/230,2)#Na
        data[5] = round(float(data[5])/391,2) #K
        data[6] = round(float(data[6])/506.47,2) #Cl
        
    return arr
