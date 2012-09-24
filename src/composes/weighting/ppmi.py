'''
Created on Sep 20, 2012

@author: georgianadinu
'''

from weighting import Weighting
from epmi import EpmiWeighting

class PpmiWeighting(Weighting):
    '''
    classdocs
    '''
    _name = "ppmi"

    def apply_weighting(self, matrix_, column_marginal = None):
        
        matrix_ = EpmiWeighting().apply_weighting(matrix_, column_marginal)
        matrix_.plog()
        return matrix_
    
    def getName(self):
        return self.__name
    
    def __str__(self):
        return self.__name