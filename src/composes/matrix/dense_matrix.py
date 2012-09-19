'''
Created on Sep 17, 2012

@author: georgianadinu
'''

import numpy as np
from warnings import warn
from scipy.sparse import issparse
from composes.matrix.matrix import Matrix

class DenseMatrix(Matrix):
    '''
    classdocs
    '''

    def __init__(self, data, *args, **kwargs):
        '''
        Constructor, creates a DenseMatrix from a numpy matrix-like
        object.
        
        Matrix-like objects (np.ndarray, np.matrix, scipy.sparse.matrix,
         SparseMatrix) are converted into np.matrix.
        
        Params:
            data: numpy matrix-like object or Matrix type
            
        Raises:
            TypeError
        '''
          
        if issparse(data):
            warn("Convert scipy sparse matrix to numpy dense matrix.")
            self.mat = data.todense()
        elif isinstance(data, np.ndarray):
            self.mat = np.matrix(data)
        elif isinstance(data, np.matrix):
            self.mat = data
        elif isinstance(data, Matrix):
            warn("Convert DenseMatrix to SparseMatrix")
            self.mat = data.toDenseMatrix().mat
        else:
            # TODO: raise suitable message
            raise TypeError("expected matrix-like type, received %s"
                            % type(data))
        
    def multiply(self, matrix_):
        '''
        Component-wise multiplication
        '''
        #TODO CHECK TYPE HERE
        if not isinstance(matrix_, DenseMatrix):
            raise TypeError("expected DenseMatrix, received %s" 
                            % (type(matrix_)))
        if self.mat.shape != matrix_.mat.shape:
            raise ValueError("inconsistent shapes: %s %s" 
                             % (str(self.mat.shape), str(matrix_.mat.shape) ))
        return DenseMatrix(np.multiply(self.mat, matrix_.mat))
    
    def toSparseMatrix(self):
        '''
        Convert SparseMatrix to DenseMatrix
        '''
        from composes.matrix.sparse_matrix import SparseMatrix
        return SparseMatrix(self.mat)
    
    def toDenseMatrix(self, copy = False):
        if (copy):
            return self.copy()
        else:
            return self    
        