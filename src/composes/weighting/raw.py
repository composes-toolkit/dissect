'''
Created on Jun 3, 2012

@author: thenghia.pham
'''

from weighting import Weighting

class RawWeighting(Weighting):
    '''
    Raw weighting scheme.
    
    This method simply returns the original matrix,
    '''
    _name = "raw"

    def apply(self, matrix_, column_marginal=None):
        return matrix_
