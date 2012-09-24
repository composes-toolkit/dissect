'''
Created on Jun 3, 2012

@author: thenghia.pham
'''
from weighting import Weighting
from composes.utils.matrix_utils import nonzero_invert

class EpmiWeighting(Weighting):
    '''
    Exponential Pointwise Mututal Information weighting
    '''
    
    _name = 'epmi'
    
    def apply_weighting(self, matrix_, column_marginal = None):
        '''
        Perform EPMI weighting
        
        Args:
            matrix_ (Matrix): Input matrix
            column_marginal (array): column marginals of the core matrix if the matrix is a peripheral matrix
    
        Returns:
            Matrix: the matrix after applying EPMI
            
        '''
        
        matrix_.assert_positive()
        row_sum = matrix_.sum(axis = 1)
        
        if not column_marginal is None:
            col_sum = column_marginal
        else:
            col_sum = matrix_.sum(axis = 0)
            
        total = col_sum.sum()
        
        row_sum = nonzero_invert(row_sum)
        col_sum = nonzero_invert(col_sum)
        col_sum = col_sum * total
        
        matrix_ = matrix_.scale_rows(row_sum)
        matrix_ = matrix_.scale_columns(col_sum)
        
        return matrix_
    
    def getName(self):
        return self.__name
    
    def __str__(self):
        return self.__name
