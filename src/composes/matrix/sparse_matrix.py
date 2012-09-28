'''
Created on Sep 17, 2012

@author: georgianadinu
'''

import numpy as np
from warnings import warn
from scipy.sparse import issparse
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from sparsesvd import sparsesvd
from composes.matrix.matrix import Matrix
from composes.utils.matrix_utils import array_to_csr_diagonal


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
        elif isinstance(data, np.matrix):
            if data.shape[0] == 0 or data.shape[1] == 0:
                raise ValueError("cannot initialize empty matrix")
            warn("Convert numpy dense array to scipy sparse matrix")
            self.mat = csr_matrix(data)
            
        elif isinstance(data, np.ndarray):
            if len(data) == 0:
                raise ValueError("cannot initialize empty matrix")
            warn("Convert numpy dense array to scipy sparse matrix")
            self.mat = csr_matrix(data)
            
        elif isinstance(data, Matrix):
            warn("Convert DenseMatrix to SparseMatrix")
            self.mat = data.to_sparse_matrix().mat
        else:
            raise TypeError("expected scipy sparse matrix, received %s" 
                            % (type(data)))
        

    def multiply(self, matrix_):
        '''
        Component-wise multiplication
        '''
        self._assert_same_type(matrix_)
        return SparseMatrix(self.mat.multiply(matrix_.mat))
        
    def svd(self, reduced_dimension):        
        '''
        '''
        #svds from scipy.sparse.linalg
        #RAISES ValueError if the rank is smaller than reduced_dimension + 1
        #TODO : fix this or replace with svdsparse
        #??? eIGENVALUES ARE NOT SORTED!!!!!!
        #IF EVER USE THIS; FIX THE PROBLEMS
        #u, s, vt = svds(self.mat, False, True)
        
        if reduced_dimension == 0:
            raise ValueError("Cannot reduce to dimensionality 0.")
        
        ut, s, vt = sparsesvd(self.mat.tocsc(), reduced_dimension)
        u = ut.transpose()
        rank = u.shape[1]
        u_matrix = SparseMatrix(u)
        vt_matrix = SparseMatrix(csr_matrix(vt.transpose()))
        
        if reduced_dimension > self.mat.shape[1]:
            warn("Number of columns smaller than the reduced dimensionality requested: %d < %d. Truncating to %d dimensions (rank)." % (self.mat.shape[1], reduced_dimension, rank))
        
        if reduced_dimension > rank:
            warn("Rank of matrix smaller than the reduced dimensionality requested: %d < %d. Truncating to %d dimensions." % (rank, reduced_dimension, rank))
        
        return u_matrix, s, vt_matrix
    
    def vstack(self, matrix_):
        self._assert_same_type(matrix_)
        return SparseMatrix(vstack([self.mat, matrix_.mat], format = "csr"))
        
    def assert_positive(self):
        if not np.all(self.mat.data >= 0):
            raise ValueError("expected non-negative matrix")
        
    def scale_rows(self, array_):
        
        self._assert_array(array_)
        
        diag_matrix = array_to_csr_diagonal(array_)
        return SparseMatrix(diag_matrix * self.mat)
        
    def scale_columns(self, array_):
        
        self._assert_array(array_)
        
        diag_matrix = array_to_csr_diagonal(array_)
        return SparseMatrix(self.mat * diag_matrix)
        
    def plog(self):
        '''
        Applies positive log to the matrix elements.
        
        Elements smaller than 1 (leading to not defined log or negative log)
        are set to 0. Log is applied on all other elements.
        '''
        
        self.mat.data[self.mat.data <= 1] = 1
        self.mat.data = np.log(self.mat.data)
        self.mat.eliminate_zeros()
            
    def to_dense_matrix(self):
        '''
        Convert to DenseMatrix.
        '''
        from composes.matrix.dense_matrix import DenseMatrix
        return DenseMatrix(self.mat)
    
    def to_sparse_matrix(self, copy = False):
        if (copy):
            return self.copy()
        else:
            return self    
    
    
     