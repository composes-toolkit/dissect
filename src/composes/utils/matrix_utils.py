
import numpy as np
from composes.matrix.sparse_matrix import SparseMatrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.matrix import Matrix
from scipy.sparse import issparse
from composes.utils.py_matrix_utils import is_array
from warnings import warn

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

def padd_matrix(matrix_, axis, value=1):
    matrix_type = type(matrix_)
    if axis == 0:
        append_mat = matrix_type(np.ones((1, matrix_.shape[1]))*value)
        return matrix_.vstack(append_mat)
    elif axis == 1:
        append_mat = matrix_type(np.ones((matrix_.shape[0], 1))*value)
        return matrix_.hstack(append_mat)
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


def to_compatible_matrix_types(v1, v2):

    if isinstance(v1, Matrix) and isinstance(v2, Matrix):
        v2 = type(v1)(v2)
    elif not isinstance(v1, Matrix) and isinstance(v2, Matrix):
        v1 = type(v2)(v1)
    elif not isinstance(v2, Matrix) and isinstance(v1, Matrix):
        v2 = type(v1)(v2)
    else:
        v1 = to_matrix(v1)
        v2 = type(v1)(v2)

    return v1, v2



def get_type_of_largest(matrix_list):
    max_dim = 0
    max_type = None
    for matrix_ in matrix_list:
        if matrix_.shape[0] * matrix_.shape[1] > max_dim:
            max_type = type(matrix_)
            max_dim = matrix_.shape[0] * matrix_.shape[1]

    return max_type

def resolve_type_conflict(matrix_list, matrix_type):
    new_matrix_list = []

    if matrix_type_conflict(matrix_list):
        warn("Efficiency warning: matrices should have the same dense/sparse type!")
        for matrix_ in matrix_list:
            new_matrix_list.append(matrix_type(matrix_))
        return new_matrix_list

    return list(matrix_list)


def matrix_type_conflict(matrix_list):

    if not matrix_list:
        return False

    matrix_type = type(matrix_list[0])
    for matrix_ in matrix_list:
        if not isinstance(matrix_, matrix_type):
            return True

    return False



