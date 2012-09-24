'''
Created on Sep 20, 2012

@author: georgianadinu
'''

from weighting import Weighting
from ppmi import PpmiWeighting

class PlmiWeighting(Weighting):
    '''
    classdocs
    '''
    _name = "plmi"

    def apply_weighting(self, matrix_, column_marginal = None):
        return matrix_.multiply(PpmiWeighting().apply_weighting(matrix_,
                                                                column_marginal))

    
    def getName(self):
        return self.__name
    
    def __str__(self):
        return self.__name