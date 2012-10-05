'''
Created on Oct 5, 2012

@author: georgianadinu
'''

from composition_model import CompositionModel

class Multiplicative(CompositionModel):
    '''
    classdocs
    '''
    _name = "multiplicative"

    def __init__(self):
        '''
        Constructor
        '''
        
    @classmethod    
    def train(cls):    
        raise NotImplementedError("Cannot train multiplicative model!")
    
    def _compose(self, arg1_mat, arg2_mat):
        return arg1_mat.multiply(arg2_mat)
            