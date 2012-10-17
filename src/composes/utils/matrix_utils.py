'''
Created on Sep 19, 2012

@author: georgianadinu
'''
import numpy as np
from scipy.sparse import spdiags


def array_to_csr_diagonal(array_):
    #array_ can't be a sparse matrix, if it is dense, it has to be a row matrix
    #(i.e. shape = (1, x))
    
    flat_array = array_.flatten()
    array_size = flat_array.size
    csr_diag = spdiags(flat_array, [0], array_size, array_size, format = 'csr')
    return csr_diag

def is_array(operand):
    return hasattr(operand, 'dtype') and hasattr(operand, 'shape')
    
    
def nonzero_invert(matrix_):
    '''
    Performs 1/x for all x, non-zero elements of the matrix.
    
    Params:
        matrix_: np.matrix 
    '''
        
    matrix_ = matrix_.astype(np.double)
    matrix_[matrix_ != 0] = np.array(1.0/matrix_[matrix_ != 0]).flatten()
    return matrix_    


