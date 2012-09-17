'''
Created on Sep 14, 2012

@author: georgianadinu
'''

if __name__ == '__main__':
    def main(): 
    
    #load saved space
    ..
    #Initialize a composition model
    
    #Option1: provide the parameters
    params = WeightedAdditiveModel.train()
    comp = WeightedAdditiveModel(params)
    
    #
    params = WeightedAdditiveModel.load("file")
    comp = WeightedAdditiveModel(params)
    
    #
    params = (1,1)
    comp = WeightedAdditiveModel(params)
    
    
    #
    comp = WeightedAdditiveModel.train("input data file", core_space, phrase_space)
    comp = WeightedAdditiveModel.load("input data file")
    comp.save("input data file")
    
    comp = WeightedAdditiveModel(params)
    
    #1
    comp = WeightedAdditiveModel()
    comp.train("input data file")
    comp.save("file")
    
    #2
    comp = WeightedAdditiveModel()
    comp.load("file")

    #3
    params = (1,1)
    comp = WeightedAdditiveModel(params)
    
    

    #Create a new composed space
    phrase_space = composition.createComposedSpace(my_space, ...)
    
    #
    calculator = SimilarityCalculator(phrase_space)
    
    print calculator.computeSimilarity("middle-age_man", "red_porsche")
    
    