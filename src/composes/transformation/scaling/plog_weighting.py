'''
Created on Jun 3, 2012

@author: thenghia.pham
'''
from scaling import Scaling
    
class PlogWeighting(Scaling):
    '''
    Positive Log Weighting
    '''
    _name = "plog"

    def apply(self, matrix_):
        '''
        Performs positive log weighting.
        
        Args:
            matrix_ (Matrix): Input matrix
            column_marginal (array): column marginals of the core matrix if the matrix is a peripheral matrix
    
        Returns:
            Matrix: the matrix after applying plog
            
        '''
        matrix_ = matrix_.copy()
        matrix_.plog()
        return matrix_
        

        