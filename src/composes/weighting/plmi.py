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

    def apply(self, matrix_, column_marginal=None):
        return matrix_.multiply(PpmiWeighting().apply(matrix_,
                                                                column_marginal))

    
