'''
Created on Oct 5, 2012

@author: georgianadinu
'''

from composition_model import CompositionModel
from composes.exception.illegal_state_error import IllegalOperationError

class Multiplicative(CompositionModel):
    '''
    classdocs
    '''
    _name = "multiplicative"

    def __init__(self):
        '''
        Constructor
        '''
        
        
    def train(self):    
        raise NotImplementedError("Cannot train multiplicative model!")
    
    def _compose(self, arg1_mat, arg2_mat):
        return arg1_mat.multiply(arg2_mat)
    
    def export(self, filename):
        raise IllegalOperationError("cannot export a Multiplicative model.")
            