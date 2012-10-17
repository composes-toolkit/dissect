'''
Created on Sep 28, 2012

@author: georgianadinu
'''

from dimensionality_reduction import DimensionalityReduction
from composes.matrix.linalg import Linalg

class Svd(DimensionalityReduction):
    '''
    classdocs
    '''
    _name = "svd"
    
    def __init__(self, reduced_dimension):
        '''
        Constructor
        '''
        super(Svd, self).__init__(reduced_dimension)
        
    def apply(self, matrix_):
        u, s, v = Linalg.svd(matrix_, self._reduced_dimension)
        return u.scale_columns(s), v
        
        
    