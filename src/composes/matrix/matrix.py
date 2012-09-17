'''
Created on Sep 17, 2012

@author: georgianadinu
'''

class Matrix(object):
    '''
    Matrix class, providing a common interface for matrix implementations.
    
    Provides a common interface for different matrix implementations
    (sparse/dense). In vector space models, a matrix is used to encode a set
    of entities such as words, phrases (rows) described in terms of contextual
    features (columns). 
    '''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
    
    
    def __add__(self, matrix_):
        ''' + operation'''
        matrix_type = type(self)
        return matrix_type(self.mat + matrix_.mat)
    
    def get_mat(self):
        return self._mat
    
    def set_mat(self, mat_):
        self._mat = mat_
        
    mat = property(get_mat, set_mat)    