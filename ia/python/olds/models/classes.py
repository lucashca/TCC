class Results:
    def __init__(self,regressor=None,score=None,rand=None):
        self.regressor = regressor
        self.score = score
        self.rand = rand
    

class RegressorModel:
    def __init__(self,params,model,data,score):
        self.params = params
        self.model = model
        self.data = data
        self.score = score

class RegressorResult:
    bestMeanR2:RegressorModel
    bestMSE:RegressorModel
    bestMeanMSE:RegressorModel
    bestR2:RegressorModel

    def __init__(self):
        pass
    
    def setBestMeanR2(self,model):
        self.bestMeanR2 = model

    def setBestMSE(self,model):
        self.bestMSE = model

    def setBestMeanMSE(self,model):
        self.bestMeanMSE = model

    def setBestR2(self,model):
        self.bestR2 = model

