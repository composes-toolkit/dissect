'''
Created on Sep 17, 2012

@author: georgianadinu
'''

import numpy as np
from composes.utils.num_utils import is_numeric
from composes.utils.matrix_utils import is_array

class Matrix(object):
    '''
    Provides a common interface for matrix implementations.
    
    Provides a common interface for different matrix implementations
    (sparse/dense). In vector space models, a matrix is used to encode
    a set of entities such as words or phrases (rows) described in terms
    of contextual features (columns). 
    '''

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
        return type(self)(-self.mat)
    
    def __mul__(self, factor):
        ''' * operation'''
        if is_numeric(factor):
            return type(self)(self.mat * factor)
        else:
            self._assert_same_type(factor)
            return type(self)(self.mat * factor.mat)
        
    def __div__(self, factor):
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

    def __getitem__(self, index):
        result = self.mat[index]
        if is_numeric(result):
            return result
        else:
            return type(self)(result)
        
    #TODO move all these asserts somewhere else
    def _assert_same_type(self, operand):
        if type(self) != type(operand):
            raise TypeError("expected matrix of type %s, received %s" %
                             (type(self), type(operand)))

    def _assert_array(self, operand):        
        if not is_array(operand):
            raise TypeError("expected array, received %s" % (type(operand)))
            
    def transpose(self):
        return type(self)(self.mat.transpose())
                    
    def sum(self, axis=None):
        #return type is dense matrix of shape (1, dimy) or (dimx,1)
        #or a number if **kwargs is None
        return self.mat.sum(axis)
            
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
    
    #TODO: implement scale_rows, scale_columns:
    #sparse: mult with diagonal
    #dense: use broadcasting
    #use them in weighting/normalization (to unify it)
    
    
            