'''
Created on Sep 14, 2012

@author: georgianadinu
'''



def main(): 
    
    #load saved space
    
    #Initialize a composition model
    
    #Option1: provide the parameters
    composition = WeightedAdditiveModel(1,1)
    

    #Create a new composed space
    phrase_space = composition.createComposedSpace(my_space, ...)
    
    #
    calculator = SimilarityCalculator(phrase_space)
    
    print calculator.computeSimilarity("middle-age_man", "red_porsche")
    
    
if __name__ == '__main__':
    pass