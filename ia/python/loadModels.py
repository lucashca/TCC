from dataSetUtils import normalizeColumn,denormalizeVal,getMax,getMin,normalizeVal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import os
import joblib
import pandas as pd
import numpy as np

class ModelsControlers:

    def __init__(self):
        csvFile = pd.read_csv("mainDataSet1.csv",usecols=[1,2,3,4,5,6])
        self.dataSet = np.array(csvFile.values)
        self.modelCa = joblib.load('./dumps/modelCa.joblib')
        self.modelMg = joblib.load('./dumps/modelMg.joblib')
        self.modelNa = joblib.load('./dumps/modelNa.joblib')
        self.modelK = joblib.load('./dumps/modelK.joblib')
        self.minLat,self.maxLat = self.getMinMax(self.dataSet,0)
        self.minLng,self.maxLng = self.getMinMax(self.dataSet,1)
        self.minCa,self.maxCa = self.getMinMax(self.dataSet,2)
        self.minMg,self.maxMg = self.getMinMax(self.dataSet,3)
        self.minNa,self.maxNa = self.getMinMax(self.dataSet,4)
        self.minK,self.maxK = self.getMinMax(self.dataSet,5)
        self.normalizeDataSet()

    def getMinMax(self,dataSet,column):
        return getMin(dataSet,column),getMax(dataSet,column)

    def normalizeDataSet(self):    
        normalizeColumn(self.dataSet,0)
        normalizeColumn(self.dataSet,1)
        normalizeColumn(self.dataSet,2)
        normalizeColumn(self.dataSet,3)
        normalizeColumn(self.dataSet,4)
        normalizeColumn(self.dataSet,5)

    
    def modelCaPredict(self,latitude,longitude):
        lat = normalizeVal(latitude,self.minLat,self.maxLat)
        lng = normalizeVal(longitude,self.minLng,self.maxLng)
        ent = np.array([lat,lng])
        res = self.modelCa.predict([ent])
        return denormalizeVal(res,self.minCa,self.maxCa)


    def modelMgPredict(self,latitude,longitude):
        lat = normalizeVal(latitude,self.minLat,self.maxLat)
        lng = normalizeVal(longitude,self.minLng,self.maxLng)
        ent = np.array([lat,lng])
        res = self.modelMg.predict([ent])
        return denormalizeVal(res,self.minMg,self.maxMg)


    def modelNaPredict(self,latitude,longitude):
        lat = normalizeVal(latitude,self.minLat,self.maxLat)
        lng = normalizeVal(longitude,self.minLng,self.maxLng)
        ent = np.array([lat,lng])
        res = self.modelNa.predict([ent])
        return denormalizeVal(res,self.minNa,self.maxNa)

    def modelKPredict(self,latitude,longitude):
        lat = normalizeVal(latitude,self.minLat,self.maxLat)
        lng = normalizeVal(longitude,self.minLng,self.maxLng)
        ent = np.array([lat,lng])
        res = self.modelK.predict([ent])
        return denormalizeVal(res,self.minK,self.maxK)

