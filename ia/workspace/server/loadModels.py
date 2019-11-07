
import sys
sys.path.insert(0, "../myTools")
from dataSetPreProcessing import normalizeColumn,denormalizeVal,getMax,getMin,normalizeVal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import os
import joblib
import pandas as pd
import numpy as np





from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from loadDataSet import loadMainDataSet, loadTesteDataSet, loadCompletDataSet, loadMainDataSetWithElevation,loadMainDataSetWithElevationWithoutNormalization
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


dataSet = loadMainDataSetWithElevation()



class ModelsControlers:

    def __init__(self):
        self.dataSet = loadMainDataSetWithElevationWithoutNormalization()
        self.modelMg = joblib.load('./dumps/modelMg.joblib')
        self.modelNa = joblib.load('./dumps/modelNa.joblib')
        self.modelK = joblib.load('./dumps/modelK.joblib')
        
        self.minLat,self.maxLat = self.getMinMax(self.dataSet,0)
        self.minLng,self.maxLng = self.getMinMax(self.dataSet,1)
        self.minElev,self.maxElev = self.getMinMax(self.dataSet,2)
        self.minCa,self.maxCa = self.getMinMax(self.dataSet,3)
        self.minMg,self.maxMg = self.getMinMax(self.dataSet,4)
        self.minNa,self.maxNa = self.getMinMax(self.dataSet,5)
        self.minK,self.maxK = self.getMinMax(self.dataSet,6)
    
    def getMinMax(self,dataSet,column):
        return getMin(dataSet,column),getMax(dataSet,column)

    def normalizeInputData(self,latitude,longitude,elevation,ca):
        lat = normalizeVal(latitude,self.minLat,self.maxLat)
        lng = normalizeVal(longitude,self.minLng,self.maxLng)
        elev = normalizeVal(elevation,self.minElev,self.maxElev)
        ca = normalizeVal(ca,self.minCa,self.maxCa)
        print(self.minLat,self.maxLat)
        inputData = np.array([lat,lng,elev,ca])
        print(inputData)
        return inputData

    def modelMgPredict(self,latitude,longitude,elevation,ca):
        inputData = self.normalizeInputData(latitude,longitude,elevation,ca)
        res = self.modelMg.predict([inputData])
        return denormalizeVal(res,self.minMg,self.maxMg)


    def modelNaPredict(self,latitude,longitude,elevation,ca):
        inputData = self.normalizeInputData(latitude,longitude,elevation,ca)
        res = self.modelNa.predict([inputData])
        return denormalizeVal(res,self.minNa,self.maxNa)

    def modelKPredict(self,latitude,longitude,elevation,ca):
        inputData = self.normalizeInputData(latitude,longitude,elevation,ca)
        res = self.modelK.predict([inputData])
        return denormalizeVal(res,self.minK,self.maxK)

