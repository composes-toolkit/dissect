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
        self
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
            if len(data) == 0:
                raise ValueError("cannot initialize empty matrix")
            self.mat = np.matrix(data)
        elif isinstance(data, np.matrix):
            if data.shape[0] == 0 or data.shape[1] == 0:
                raise ValueError("cannot initialize empty matrix")
            self.mat = data
        elif isinstance(data, Matrix):
            warn("Convert DenseMatrix to SparseMatrix")
            self.mat = data.to_dense_matrix().mat
        else:
            # TODO: raise suitable message
            raise TypeError("expected matrix-like type, received %s"
                            % type(data))
        
    def multiply(self, matrix_):
        '''
        Component-wise multiplication
        '''
        self._assert_same_type(matrix_)
        if self.mat.shape != matrix_.mat.shape:
            raise ValueError("inconsistent shapes: %s %s" 
                             % (str(self.mat.shape), str(matrix_.mat.shape) ))
        return DenseMatrix(np.multiply(self.mat, matrix_.mat))
    
    def vstack(self, matrix_):
        self._assert_same_type(matrix_)
        return DenseMatrix(np.vstack((self.mat, matrix_.mat)))
    
    def svd(self, reduced_dimension):
        '''
           - return three outputs
            + u: u matrix
            + s: flat version of s matrix
            + vt: transpose of v matrix
        '''
        if reduced_dimension == 0:
            raise ValueError("Cannot reduce to dimensionality 0.")
        u, s, vt = np.linalg.svd(self.mat, False, True)
        tol = 1e-12
        rank = len(s[s > tol])
        
        if reduced_dimension > self.mat.shape[1]:
            warn("Number of columns smaller than the reduced dimensionality requested: %d < %d. Truncating to %d dimensions (rank)." % (self.mat.shape[1], reduced_dimension, rank))
        elif reduced_dimension > rank:
            warn("Rank of matrix smaller than the reduced dimensionality requested: %d < %d. Truncating to %d dimensions." % (rank, reduced_dimension, rank))
                    
        no_cols = min(rank, reduced_dimension)
        u = u[:,0:no_cols]
        s = s[0:no_cols]
        vt = vt[0:no_cols,:]

        return DenseMatrix(u), s, DenseMatrix(vt.transpose())
            
    def scale_rows(self, array_):
        '''
        Scales rows by elements in array.
        '''
        #TODO maybe return a copy here and not destroy the original??
        self._assert_array(array_)
       
        x_dim = self.mat.shape[0]
        if array_.shape in ((x_dim, 1), (x_dim,)):
            if array_.shape == (x_dim,):
                array_ = array_.reshape((x_dim, 1))
            return DenseMatrix(np.multiply(self.mat, array_))
        else:
            raise ValueError("inconsistent shapes: %s %s"
                             % (str(self.mat.shape), str(array_.shape)))    
        
    def scale_columns(self, array_):
        '''
        Scales columns by elements in array.
        '''
        #TODO maybe return a copy here and not destroy the original??
        self._assert_array(array_)
                    
        y_dim = self.mat.shape[1]
        if array_.shape in ((1, y_dim), (y_dim,)):
            return DenseMatrix(np.multiply(self.mat, array_))
        else:
            raise ValueError("inconsistent shapes: %s %s"
                             % (str(self.mat.shape), str(array_.shape)))    
    
    def plog(self):
        '''
        Applies positive log to the matrix elements.
        
        Elements smaller than 1 (leading to not-defined log or negative log)
        are set to 0. Log is applied on all other elements.
        '''
        
        #this line uses 3 x size(mat) to run in the worst case
        #(if we select the entire matrix - depends on the size of the selection)
        self.mat[self.mat < 1.0] = 1
        self.mat = np.log(self.mat)    
    
    
    def assert_positive(self):
        if not np.all(self.mat >= 0):
            raise ValueError("expected non-negative matrix") 
                
    def to_sparse_matrix(self):
        '''
        Convert SparseMatrix to DenseMatrix
        '''
        from composes.matrix.sparse_matrix import SparseMatrix
        return SparseMatrix(self.mat)
    
    def to_dense_matrix(self, copy = False):
        if (copy):
            return self.copy()
        else:
            return self    
        