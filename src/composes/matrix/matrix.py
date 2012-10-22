'''
Created on Sep 17, 2012

@author: georgianadinu
'''

from composes.utils.num_utils import is_numeric
from composes.utils.py_matrix_utils import is_array

class Matrix(object):
    """
    Provides a common interface for matrix implementations.
    
    Provides a common interface for different matrix implementations
    (sparse/dense). In vector space models, a matrix is used to encode
    a set of entities such as words or phrases (rows) described in terms
    of contextual features (columns). 
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    def __add__(self, matrix_):
        ''' + operation'''
        self._assert_same_type(matrix_)
        return type(self)(self.mat + matrix_.mat)
    
    def __sub__(self, matrix_):
        ''' - operation'''
        self._assert_same_type(matrix_)
        return type(self)(self.mat - matrix_.mat)
    
    def __neg__(self):
        ''' - operation'''
        return type(self)(-self.mat)
    
    def __mul__(self, factor):
        ''' * operation'''
        if is_numeric(factor):
            return type(self)(self.mat * factor)
        else:
            self._assert_same_type(factor)
            return type(self)(self.mat * factor.mat)
        
    def __div__(self, factor):
        ''' / operation'''
        if is_numeric(factor):
            if factor == 0:
                raise ZeroDivisionError("Division by zero")
        else:
            raise TypeError("expected numeric type, received %s" % (type(factor)))
        return type(self)(self.mat / float(factor))    
                 
    def __rmul__(self, factor):
        ''' * operation'''
        if is_numeric(factor):
            return self.__mul__(factor)
        raise TypeError("expected numeric type, received %s" % (type(factor)))

        
    #TODO move all these asserts somewhere else
    def _assert_same_type(self, operand):
        if type(self) != type(operand):
            raise TypeError("expected matrix of type %s, received %s" %
                             (type(self), type(operand)))
    
    def assert_same_shape(self, matrix_):
        """
        Asserts that the matrix has the same shape as a second matrix.
        
        Args:
            matrix_: A second matrix of type Matrix.
            
        Raises:
            ValueError: If the current matrix and the argument matrix 
                do not have the same shape.
        """
        
        if self.mat.shape != matrix_.mat.shape:
            raise ValueError("inconsistent shapes: %s %s" 
                             % (str(self.mat.shape), str(matrix_.mat.shape) ))
            
    #TODO move all these asserts somewhere else
    def _assert_array(self, operand):        
        if not is_array(operand):
            raise TypeError("expected array, received %s" % (type(operand)))

                    
    def sum(self, axis=None):
        #return type is dense matrix of shape (1, dimy) or (dimx,1)
        #or a number if **kwargs is None
        return self.mat.sum(axis)
            
    def sorted_permutation(self, norm_function, axis_):
        """
        Computes the permutation resulted when sorting the matrix 
        on an axis, according to a function, in descending order.
            
        Sorts the rows or the columns (as given by axis)
        of a matrix according to a norm_function and returns 
        the permutation of this as a np.array
    
        Args:
            norm_function: One of sum/length. A function that 
                takes an axis as an argument (i.e. 0 or 1) and 
                returns an array of values (i.e. sum of all rows 
                if axis = 0 and norm_function = sum). 
                
            axis_: axis value, one of 0/1
            
        Returns:
            perm_srtd: np.array containing the permutation of the 
                sorting
        """    
        
        #norms = norm_function(axis=axis_)
        
        norms = norm_function(axis_).getA().flatten()
        perm_srtd = sorted(range(len(norms)), key = norms.__getitem__, 
                           reverse=True)
        
        return perm_srtd    
            
    def get_mat(self):
        return self._mat
    
    def set_mat(self, mat_):
        self._mat = mat_
    
    def get_shape(self):
        return self.mat.shape
        
    def copy(self):
        return type(self)(self.mat.copy())
        
    mat = property(get_mat, set_mat)
    shape = property(get_shape)
    
    
    
            