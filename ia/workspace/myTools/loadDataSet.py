

import sys
sys.path.insert(0, "../datasets")

from dataSetPreProcessing import normalizeColumn, deNormalizeColumn

import numpy as np
import pandas as pd



def loadMainDataSetWithElevation():

    csvFile = pd.read_csv("../datasets/DataSetWithElevation.csv", usecols=[1, 2, 3, 4, 5, 6,7])
    print(csvFile.head(2))
    dataSet = np.array(csvFile.values)
    
    #dataSet[:, 0:2] = dataSet[:, 0:2]*-1
    normalizeColumn(dataSet, 0)
    normalizeColumn(dataSet, 1)
    normalizeColumn(dataSet, 2)
    normalizeColumn(dataSet, 3)
    normalizeColumn(dataSet, 4)
    normalizeColumn(dataSet, 5)
    normalizeColumn(dataSet, 6)
    return dataSet


def loadMainDataSet():

    csvFile = pd.read_csv("../datasets/mainDataSet1.csv", usecols=[1, 2, 3, 4, 5, 6])
    print(csvFile.head(2))
    
    dataSet = np.array(csvFile.values)
#    dataSet[:, 0:2] = dataSet[:, 0:2]*-1
    normalizeColumn(dataSet, 0)
    normalizeColumn(dataSet, 1)
    normalizeColumn(dataSet, 2)
    normalizeColumn(dataSet, 3)
    normalizeColumn(dataSet, 4)
    normalizeColumn(dataSet, 5)
    return dataSet

def loadCompletDataSet():

    csvFile = pd.read_csv("../datasets/completDataset.csv", usecols=[1, 2, 3, 4, 5, 6])
    print(csvFile.head(2))
    
    dataSet = np.array(csvFile.values)
#    dataSet[:, 0:2] = dataSet[:, 0:2]*-1
    normalizeColumn(dataSet, 0)
    normalizeColumn(dataSet, 1)
    normalizeColumn(dataSet, 2)
    normalizeColumn(dataSet, 3)
    normalizeColumn(dataSet, 4)
    normalizeColumn(dataSet, 5)
    return dataSet



def loadTesteDataSet():
    csvFile = pd.read_csv("../datasets/sela.csv", usecols=[0, 1, 2])
    dataSet = np.array(csvFile.values)
    normalizeColumn(dataSet, 0)
    normalizeColumn(dataSet, 1)
    normalizeColumn(dataSet, 2)
    return dataSet
