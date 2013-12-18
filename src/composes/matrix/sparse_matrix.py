'''
Created on Sep 17, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

import numpy as np
from warnings import warn
from scipy.sparse import issparse
from scipy.sparse import vstack
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.sparse.sputils import isintlike
from composes.utils.num_utils import is_numeric
from composes.utils.num_utils import is_integer
from composes.matrix.matrix import Matrix
from composes.utils.py_matrix_utils import array_to_csr_diagonal
from scipy.sparse import identity

class SparseMatrix(Matrix):
    '''
    classdocs
    '''


    def __init__(self, data):
        """
        Constructor, creates a SparseMatrix from a numpy matrix-like
        object.

        Matrix-like objects (np.ndarray, np.matrix, scipy.sparse.matrix,
         DenseMatrix) are converted into scipy.csr_matrix.

        Args:
            data: numpy matrix-like object or Matrix type

        Raises:
            TypeError: if input data is not one of scipy.sparse/
            numpy.ndarray/numpy.matrix/Matrix
            ValueError: if trying to initialize shape-0 matrix
        """
        if issparse(data):
            self.mat = data.tocsr()

        elif isinstance(data, np.matrix):
            if data.shape[0] == 0 or data.shape[1] == 0:
                raise ValueError("cannot initialize matrix with shape 0")
            self.mat = csr_matrix(data)

        elif isinstance(data, np.ndarray):
            if len(data) == 0:
                raise ValueError("cannot initialize matrix with shape 0")
            self.mat = csr_matrix(data)

        elif isinstance(data, Matrix):
            self.mat = data.to_sparse_matrix().mat
        else:
            raise TypeError("expected scipy sparse matrix, received %s"
                            % (type(data)))

    def __str__(self):
        return str(self.mat.todense())

    def __getitem__(self, key):
        """
        Overwrites csr_matrix m[i,:], m[i] operations which are faulty in
        current scipy.sparse releases.

        """
        def __get_row(row):
            start = self.mat.indptr[row]
            end = self.mat.indptr[row + 1]
            return SparseMatrix(csr_matrix((self.mat.data[start:end],
                                            self.mat.indices[start:end],
                                            [0, end - start]),
                                           shape=(1, self.mat.shape[1]),
                                           copy=True))

        if isinstance(key, tuple):
            row = key[0]
            col = key[1]
            if isintlike(row) and row >= 0 and isinstance(col, slice):
                if col == slice(None, None, None):
                    return __get_row(row)

        if isintlike(key) and key >= 0:
            return __get_row(key)

        result = self.mat[key]
        if is_numeric(result):
            return result
        else:
            return SparseMatrix(result)

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

        if not isinstance(new_shape, tuple) or len(new_shape) != 2:
            raise ValueError("shape must be integer pair")

        no_rows, no_cols = self.mat.shape
        new_no_rows, new_no_cols = new_shape

        if not is_integer(new_no_rows) or not is_integer(new_no_cols):
            raise ValueError("shape must be integer pair")
        if no_rows * no_cols != new_no_rows * new_no_cols:
            raise ValueError("total size of new matrix must be unchanged.")

        #TODO: change here if we want a copy!!
        mat = self.mat.tocoo(copy=False)

        #upcast mat.row and mat.col
        if no_rows * no_cols >=  2**31-1:
            linear_pos = np.array(mat.row, dtype=np.int64) * no_cols + mat.col
        else:
            linear_pos = mat.row * no_cols + mat.col

        mat.row = linear_pos // new_no_cols
        mat.col = linear_pos - (mat.row * new_no_cols)

        #NOTE: change here if we want a copy!!
        self.mat = csr_matrix((mat.data, (mat.row, mat.col)), shape=new_shape)

    @staticmethod
    def identity(size):
        """
        Builds the identity matrix.

        Args:
            size: integer, the result matrix is of shape size x size

        Returns:
            Identity SparseMatrix.
        """
        # TODO: should do system-wise
        return SparseMatrix(identity(size, dtype = np.double, format = "csr"))

    def transpose(self):
        """
        Transposes the current matrix.

        Returns:
            SparseMatrix, a transpose of the current matrix.

        """
        return type(self)(self.mat.transpose())

    def multiply(self, matrix_):
        """
        Computes component-wise multiplication of two matrices.

        Args:
            matrix_: a second matrix of type SparseMatrix

        Returns:
            A SparseMatrix containing the cw multiplication of the two.

        Raises:
            TypeError: if the argument is not of type SparseMatrix
            ValueError: if the two matrices don t have the same shape.
        """
        self._assert_same_type(matrix_)
        if self.mat.shape != matrix_.mat.shape:
            raise ValueError("inconsistent shapes: %s %s"
                             % (str(self.mat.shape), str(matrix_.mat.shape) ))

        return SparseMatrix(self.mat.multiply(matrix_.mat))

    def vstack(self, matrix_):
        """
        Vertical stack of two matrices.

        Args:
            matrix_: a second matrix of type SparseMatrix

        Returns:
            A SparseMatrix, vertical stack of the two matrices.

        Raises:
            TypeError: if the argument is not of type SparseMatrix

        """
        self._assert_same_type(matrix_)
        return SparseMatrix(vstack([self.mat, matrix_.mat], format = "csr"))


    def hstack(self, matrix_):
        """
        Horizontal stack of two matrices.

        Args:
            matrix_: a second matrix of type SparseMatrix

        Returns:
            A SparseMatrix, horizontal stack of the two matrices.

        Raises:
            TypeError: if the argument is not of type SparseMatrix

        """
        self._assert_same_type(matrix_)
        return SparseMatrix(hstack([self.mat, matrix_.mat], format = "csr"))


    @classmethod
    def nary_vstack(cls, mat_list):
        """
        Class method, vertical stack of n matrices.

        Args:
            mat_list: a list of matrices of type SparseMatrix

        Returns:
            A SparseMatrix, vertical stack of the arguments.

        """
        np_mat_list = [matrix_.mat for matrix_ in mat_list]
        return SparseMatrix(vstack(np_mat_list))

    @classmethod
    def nary_hstack(cls, mat_list):
        """
        Class method, horizontal stack of n matrices.

        Args:
            mat_list: a list of matrices of type SparseMatrix

        Returns:
            A SparseMatrix, horizontal stack of the arguments.

        """

        np_mat_list = [matrix_.mat for matrix_ in mat_list]
        return SparseMatrix(hstack(np_mat_list))

    def scale_rows(self, array_):
        """
        Scales each row of the matrix by the values given in an array.

        Args:
            array_: ndarray containing the values to scale by

        Returns:
            A new SparseMatrix with scaled rows.
        """

        self._assert_array(array_)

        diag_matrix = array_to_csr_diagonal(array_)
        return SparseMatrix(diag_matrix * self.mat)

    def scale_columns(self, array_):
        """
        Scales each column of the matrix by the values given in an array.

        Args:
            array_: ndarray containing the values to scale by

        Returns:
            A new SparseMatrix with scaled columns.
        """
        self._assert_array(array_)

        diag_matrix = array_to_csr_diagonal(array_)
        return SparseMatrix(self.mat * diag_matrix)

    def plog(self):
        """
        Applies positive log to the matrix elements.

        Elements smaller than 1 (leading to not-defined log or negative log)
        are set to 0. Log is applied on all other elements.

        Modifies the current matrix.
        """

        self.mat.data[self.mat.data <= 1] = 1
        self.mat.data = np.log(self.mat.data)
        self.mat.eliminate_zeros()

    def get_non_negative(self):
        """
        Turns negative entries to 0.

        Returns:
            A new SparseMatrix matrix in which negative entries are set to 0.

        """
        mat_ = self.mat.copy()
        #TODO: time against : mat_.data[mat_.data < 0] = 0
        mat_.data = np.where(mat_.data > 0, mat_.data, 0)
        mat_.eliminate_zeros()
        return SparseMatrix(mat_)

    def to_non_negative(self):
        """
        Turns negative entries to 0.

        Modifies the current matrix: all negative entries are set to 0.

        """
        self.mat.data.clip(0, out=self.mat.data)
        self.mat.eliminate_zeros()

    def to_ones(self):
        """
        Turns strictly positive entries to 1 and negative entries to 0.

        Modifies the current matrix: all strictly positive entries are
            set to 1, all negative entries are set to 0.

        """
        self.mat.data = np.where(self.mat.data > 0, 1, 0)
        self.mat.eliminate_zeros()

    def remove_small_values(self, epsilon):
        """
        Sets values smaller than an epsilon to 0.

        Args:
            epsilon: scalar, threshold
        Returns:
            A SparseMatrix in which all values smaller than epsilon are
                set to 0.

        """
        mat_ = self.mat.copy()
        mat_.data = np.where(mat_.data > epsilon, mat_.data, 0)
        mat_.eliminate_zeros()
        return SparseMatrix(mat_)

    def assert_positive(self):
        """
        Asserts that all values are larger or equal to 0.

        Raises:
            ValueError if not all values are >= 0.
        """
        if not np.all(self.mat.data >= 0):
            raise ValueError("expected non-negative matrix")

    def is_mostly_positive(self):
        """
        Checks if more than 50% of the non zero elements of a
        matrix are positive.

        """
        return self.mat.data[self.mat.data > 0].size > self.mat.data.size/2

    def all_close(self, matrix_):
        """
        Checks of the values in two matrices are all_close.

        Args:
            matrix_: input matrix of type SparseMatrix

        Returns:
            bool: True if the elements are allclose (using np.allclose).

        """
        diff = self.mat - matrix_.mat
        return np.allclose(diff.data, np.zeros(len(diff.data)))

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
            return np.linalg.norm(self.mat.data)
        else:
            return np.sqrt(self.multiply(self).sum(axis))

    def to_dense_matrix(self):
        """
        Converts to DenseMatrix.
        """
        from composes.matrix.dense_matrix import DenseMatrix
        return DenseMatrix(self.mat)

    def to_sparse_matrix(self, copy = False):
        """
        Returns a copy is copy=True, returns self otherwise.
        """
        if (copy):
            return self.copy()
        else:
            return self





