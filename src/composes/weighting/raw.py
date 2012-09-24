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

    def apply_weighting(self, matrix_, column_marginal=None):
        return matrix_
    def get_name(self):
        return self.__name
    def __str__(self):
        return self.__name