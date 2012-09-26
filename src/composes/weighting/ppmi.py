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

    def apply(self, matrix_, column_marginal=None):
        
        matrix_ = EpmiWeighting().apply(matrix_, column_marginal)
        matrix_.plog()
        return matrix_
    
