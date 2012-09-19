'''
Created on Sep 17, 2012

@author: georgianadinu
'''

import numpy as np
from composes.utils.num_utils import isNumeric

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
        self.__checkType(matrix_)
        return type(self)(self.mat + matrix_.mat)
    
    def __sub__(self, matrix_):
        ''' - operation'''
        self.__checkType(matrix_)
        return type(self)(self.mat - matrix_.mat)
    
    def __mul__(self, factor):
        ''' * operation'''
        if isNumeric(factor):
            return type(self)(self.mat * factor)
        else:
            self.__checkType(factor)
            return type(self)(self.mat * factor.mat)
        
    def __div__(self, factor):
        if isNumeric(factor):
            if factor == 0:
                raise ZeroDivisionError("Division by zero")
        else:
            raise TypeError("expected numeric type, received %s" % (type(factor)))
        return type(self)(self.mat / float(factor))    
                 
    def __rmul__(self, factor):
        ''' * operation'''
        if isNumeric(factor):
            return self.__mul__(factor)
        raise TypeError("expected numeric type, received %s" % (type(factor)))

    def __checkType(self, operand):
        if type(self) != type(operand):
            raise TypeError("expected matrix of type %s, received %s" %
                             (type(self), type(operand)))

        
    def get_mat(self):
        return self._mat
    
    def set_mat(self, mat_):
        self._mat = mat_
        
    mat = property(get_mat, set_mat)
    
    #TODO: implement scale_rows, scale_columns:
    #sparse: mult with diagonal
    #dense: use broadcasting
    #use them in weighting/normalization (to unify it)
    
    
            