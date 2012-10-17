'''
Created on Jun 3, 2012

@author: thenghia.pham
'''

from scaling import Scaling

class RawWeighting(Scaling):
    '''
    Raw weighting scheme.
    
    This method simply returns the original matrix,
    '''
    _name = "raw"

    def apply(self, matrix_):
        return matrix_
