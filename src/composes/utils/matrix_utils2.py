
import numpy as np
from composes.matrix.sparse_matrix import SparseMatrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.matrix import Matrix
from scipy.sparse import issparse
from matrix_utils import is_array


def to_matrix(matrix_):
    """
    Converts an array-like structure to a DenseMatrix/SparseMatrix
    """
    if issparse(matrix_):
        return SparseMatrix(matrix_)
    else:
        return DenseMatrix(matrix_)
    
def is_array_or_matrix(data):
    return is_array(data) or isinstance(data, Matrix)


def assert_is_array_or_matrix(data): 
    if not is_array_or_matrix(data):
        raise TypeError("expected array-like or matrix, received %s" 
                        % (type(data)))
        
def padd_ones(matrix_, axis):
    matrix_type = type(matrix_)
    if axis == 0:  
        ones_mat = matrix_type(np.ones((1, matrix_.shape[1])))
        return matrix_.vstack(ones_mat)
    elif axis == 1:
        ones_mat = matrix_type(np.ones((matrix_.shape[0], 1)))
        return matrix_.hstack(ones_mat)
    else:
        raise ValueError("Invalid axis value:%s" % axis)
          
          
def assert_same_shape(matrix1, matrix2, axis=None):
   
    if axis is None:
        if matrix1.shape != matrix2.shape:
            raise ValueError("Inconsistent shapes")
    else:
        if not axis in [0, 1]:
            raise ValueError("Invalid axis value: %s, expected 0 or 1." % axis)
        if matrix1.shape[axis] != matrix2.shape[axis]:
            raise ValueError("Inconsistent shapes")     
        
