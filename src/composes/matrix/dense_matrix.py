'''
Created on Sep 17, 2012

@author: georgianadinu
'''

import numpy as np
from warnings import warn
from scipy.sparse import issparse
from composes.utils.num_utils import is_numeric
from composes.matrix.matrix import Matrix

class DenseMatrix(Matrix):
    '''
    classdocs
    '''

    def __init__(self, data, *args, **kwargs):
        """
        Constructor, creates a DenseMatrix from a numpy matrix-like
        object.

        Matrix-like objects (np.ndarray, np.matrix, scipy.sparse.matrix,
         SparseMatrix) are converted into np.matrix.
        
        Params:
            data: numpy matrix-like object or Matrix type
            
        Raises:
            TypeError: if input data is not one of scipy.sparse/
                numpy.ndarray/numpy.matrix/Matrix
        """
          
        if issparse(data):
            self.mat = data.todense()
        elif isinstance(data, np.matrix):
            if data.shape[0] == 0 or data.shape[1] == 0:
                raise ValueError("cannot initialize empty matrix")
            self.mat = data
        elif isinstance(data, np.ndarray):
            if len(data) == 0:
                raise ValueError("cannot initialize empty matrix")
            self.mat = np.matrix(data)
        elif isinstance(data, Matrix):
            # TODO: remove warning or remove import somehow fix this!!
            from composes.matrix.sparse_matrix import SparseMatrix
            self.mat = data.to_dense_matrix().mat
        else:
            # TODO: raise suitable message
            raise TypeError("expected matrix-like type, received %s"
                            % type(data))

    def __str__(self):
        return str(self.mat)
     
    def __getitem__(self, index):
        result = self.mat[index]
        if is_numeric(result):
            return result
        else:
            return type(self)(result.copy())
               
    def multiply(self, matrix_):
        """
        Computes component-wise multiplication of two matrices.
        
        Args:
            matrix_: a second matrix of type DenseMatrix
            
        Returns:
            A DenseMatrix containing the cw multiplication of the two.
            
        Raises:
            TypeError: if the argument is not of type DenseMatrix
            ValueError: if the two matrices don t have the same shape.
        """
        
        self._assert_same_type(matrix_)
        if self.mat.shape != matrix_.mat.shape:
            raise ValueError("inconsistent shapes: %s %s" 
                             % (str(self.mat.shape), str(matrix_.mat.shape) ))
        return DenseMatrix(np.multiply(self.mat, matrix_.mat))
    
    def transpose(self):
        """
        Transposes the current matrix.
        
        Returns:
            DenseMatrix, a transpose of the current matrix.
                
        """
        return type(self)(self.mat.transpose().copy())
    
    def reshape(self, new_shape):
        """
        Reshapes current matrix.
        
        Overwrites the current matrix with a new matrix of the 
        given shape!
        
        Args: 
            shape: length 2 tuple or pair of integers
        
        Raises:
            ValueError: if shape is not an integer pair or
                if new shape is inconsistent with the total
                size of the current matrix.
        """

        # TODO: change this is necessary to make a copy
        self.mat = self.mat.reshape(new_shape)
        
    @staticmethod
    def identity(size):
        """
        Builds the identity matrix.
        
        Args:
            size: integer, the result matrix is of shape size x size
            
        Returns:
            Identity DenseMatrix.
        """
        return DenseMatrix(np.eye(size, size, 0, np.double))
    
    def vstack(self, matrix_):
        """
        Vertical stack of two matrices.
        
        Args:
            matrix_: a second matrix of type DenseMatrix

        Returns:
            A DenseMatrix, vertical stack of the two matrices.

        Raises:
            TypeError: if the argument is not of type DenseMatrix            
             
        """
        self._assert_same_type(matrix_)
        return DenseMatrix(np.vstack((self.mat, matrix_.mat)))

    def hstack(self, matrix_):
        """
        Horizontal stack of two matrices.
        
        Args:
            matrix_: a second matrix of type DenseMatrix

        Returns:
            A DenseMatrix, horizontal stack of the two matrices.

        Raises:
            TypeError: if the argument is not of type DenseMatrix            
             
        """          
        self._assert_same_type(matrix_)
        return DenseMatrix(np.hstack((self.mat, matrix_.mat)))
     
    @classmethod
    def nary_vstack(cls, mat_list):
        """
        Class method, vertical stack of n matrices.
        
        Args:
            mat_list: a list of matrices of type DenseMatrix

        Returns:
            A DenseMatrix, vertical stack of the arguments.

        """
        np_mat_list = [matrix_.mat for matrix_ in mat_list]
        return DenseMatrix(np.vstack(np_mat_list))
    
    @classmethod
    def nary_hstack(cls, mat_list):
        """
        Class method, horizontal stack of n matrices.
        
        Args:
            mat_list: a list of matrices of type DenseMatrix

        Returns:
            A DenseMatrix, horizontal stack of the arguments.

        """
        
        np_mat_list = [matrix_.mat for matrix_ in mat_list]
        return DenseMatrix(np.hstack(np_mat_list))
    

       
    def scale_rows(self, array_):
        """
        Scales each row of the matrix by the values given in an array.
        
        Args:
            array_: ndarray containing the values to scale by
            
        Returns:
            A new DenseMatrix with scaled rows. 
        """
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
        """
        Scales each column of the matrix by the values given in an array.
        
        Args:
            array_: ndarray containing the values to scale by
            
        Returns:
            A new DenseMatrix with scaled columns. 
        """
        self._assert_array(array_)
                    
        y_dim = self.mat.shape[1]
        if array_.shape in ((1, y_dim), (y_dim,)):
            return DenseMatrix(np.multiply(self.mat, array_))
        else:
            raise ValueError("inconsistent shapes: %s %s"
                             % (str(self.mat.shape), str(array_.shape)))    
    
    def plog(self):
        """
        Applies positive log to the matrix elements.
        
        Elements smaller than 1 (leading to not-defined log or negative log)
        are set to 0. Log is applied on all other elements.
        
        Modifies the current matrix. 
        """
        
        #this line uses 3 x size(mat) to run in the worst case
        #(if we select the entire matrix - depends on the size of the selection)
        self.mat[self.mat < 1.0] = 1
        self.mat = np.log(self.mat)    
    
    
    def assert_positive(self):
        """
        Asserts that all values are larger or equal to 0.
        
        Raises:
            ValueError if not all values are >= 0.
        """        
        if not np.all(self.mat >= 0):
            raise ValueError("expected non-negative matrix") 

    def get_non_negative(self):
        """
        Turns negative entries to 0.
        
        Returns:
            A new DenseMatrix matrix in which negative entries are set to 0.
            
        """
        mat_ = self.mat.copy()
        # TODO: time against : mat_.data[mat_.data < 0] = 0
        mat_ = np.where(mat_ > 0, mat_, 0)
        return DenseMatrix(mat_)
 
    def to_non_negative(self):
        """
        Turns negative entries to 0.
        
        Modifies the current matrix: all negative entries are set to 0.
            
        """
        
        self.mat = np.where(self.mat > 0, self.mat, 0)
    
    def to_ones(self):
        """
        Turns strictly positive entries to 1 and negative entries to 0.
        
        Modifies the current matrix: all strictly positive entries are
            set to 1, all negative entries are set to 0.
            
        """
        
        self.mat = np.where(self.mat > 0, 1, 0)
        
    def remove_small_values(self, epsilon):
        """
        Sets values smaller than an epsilon to 0.
        
        Args:
            epsilon: scalar, threshold
        Returns:
            A DenseMatrix in which all values smaller than epsilon are
                set to 0.
           
        """
        mat_ = self.mat.copy()
        mat_ = np.where(mat_ > epsilon, mat_, 0)
        return DenseMatrix(mat_)
        
    def is_mostly_positive(self):
        """
        Checks if more than 50% of the non zero elements of a 
        matrix are positive.
        
        """        
        return self.mat[self.mat > 0].size > self.mat.size/2 

    def all_close(self, matrix_):
        """
        Checks of the values in two matrices are all_close.
        
        Args:
            matrix_: input matrix of type DenseMatrix
            
        Returns:
            bool: True if the elements are allclose (using np.allclose).
            
        """
        return np.allclose(self.mat, matrix_.mat)

    def norm(self, axis = None):
        """
        Computes the norms on a certain axis or of the entire matrix.
        
        Args:
            axis: 0/1 or None, if axis is None computes the norm of the 
                full matrix
        Returns:
            nd.array containing the norms on a given axis, or a scalar
            if the axis is None.
            
        """        
        if axis is None:
            return np.linalg.norm(self.mat)
        else:
            return np.sqrt(self.multiply(self).sum(axis))
    
    def to_sparse_matrix(self):
        """
        Converts to SparseMatrix.
        """
        from composes.matrix.sparse_matrix import SparseMatrix
        return SparseMatrix(self.mat)
    
    def to_dense_matrix(self, copy = False):
        """
        Returns a copy is copy=True, returns self otherwise.
        """
        
        if (copy):
            return self.copy()
        else:
            return self    
        
