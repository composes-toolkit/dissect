'''
Created on Sep 17, 2012

@author: georgianadinu
'''

import numpy as np
from warnings import warn
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from composes.matrix.matrix import Matrix



class SparseMatrix(Matrix):
    '''
    classdocs
    '''


    def __init__(self, data, *args, **kwargs):
        '''
        Constructor, creates a SparseMatrix from a numpy matrix-like
        object.
        
        Matrix-like objects (np.ndarray, np.matrix, scipy.sparse.matrix,
         DenseMatrix) are converted into scipy.csr_matrix.
        
        Params:
            data: numpy matrix-like object or Matrix type
            
        Raises:
            TypeError
        '''
        if issparse(data):
            self.mat = data.tocsr()
        elif isinstance(data, np.matrix) or isinstance(data, np.ndarray):
            warn("Convert numpy dense matrix/array to scipy sparse matrix")
            self.mat = csr_matrix(data)
        elif isinstance(data, Matrix):
            warn("Convert DenseMatrix to SparseMatrix")
            self.mat = data.toSparseMatrix().mat
        else:
            raise TypeError("expected scipy sparse matrix, received %s" 
                            % (type(data)))
        

    def multiply(self, matrix_):
        '''
        Component-wise multiplication
        '''
        #TODO check type here
        if not isinstance(matrix_, SparseMatrix):
            raise TypeError("expected SparseMatrix, received %s" 
                            % (type(matrix_)))
        
        return SparseMatrix(self.mat.multiply(matrix_.mat))
            
        
    def toDenseMatrix(self):
        '''
        Convert to DenseMatrix.
        '''
        from composes.matrix.dense_matrix import DenseMatrix
        return DenseMatrix(self.mat)
    
    def toSparseMatrix(self, copy = False):
        if (copy):
            return self.copy()
        else:
            return self    
    
    
     