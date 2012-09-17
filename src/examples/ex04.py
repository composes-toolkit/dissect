'''
Created on Sep 14, 2012

@author: georgianadinu
'''

if __name__ == '__main__':
    '''
Created on Sep 14, 2012

@author: georgianadinu
'''



def main(): 
    
    #load saved space
    
    #Initialize a composition  model
        
    #Option2: Load a pretrained model
    paras = load(filename)
    composition = WeightedAdditiveModel(paras)
 
    #compose two words and return the vector
    red_car_vector = composition.compose(my_space, ("red", "car"))
    man_vector = my_space("man")
    
    #
    print similarity.computeSimilarity(red_car_vector, man_vector)
    
        
    
    
if __name__ == '__main__':
    pass