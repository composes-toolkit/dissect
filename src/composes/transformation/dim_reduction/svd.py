'''
Created on Sep 28, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

from dimensionality_reduction import DimensionalityReduction
from composes.matrix.linalg import Linalg

class Svd(DimensionalityReduction):
    """
    Performs truncated Singular Value Decomposition to a reduced dimension :math:`k`.
    
    Given an input matrix :math:`X`, it computes the decomposition:
    
    :math:`X = U \\Sigma V^{T}`
    
    It returns :math:`U \\Sigma` truncated to dimension  :math:`min(k,rank(X))` 
    """
    
    _name = "svd"
    
    def __init__(self, reduced_dimension):
        '''
        Constructor
        '''
        super(Svd, self).__init__(reduced_dimension)
        
    def apply(self, matrix_):

        u, s, v = Linalg.svd(matrix_, self._reduced_dimension)
        return u.scale_columns(s), v
        
    