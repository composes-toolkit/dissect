'''
Created on Oct 2, 2012

@author: georgianadinu
'''
import numpy as np
from composes.utils.matrix_utils import is_array
from composes.utils.matrix_utils2 import to_matrix
from composes.utils.matrix_utils2 import assert_is_array_or_matrix
#from composes.utils.matrix_utils2 import assert_same_shape
from composes.matrix.matrix import Matrix


class Similarity(object):
    '''
    classdocs
    '''

    _name = "we are NOT stupid"

    def __init__(self):
        '''
        Constructor
        '''
        pass
 
    def get_sim(self, v1, v2):
        
        assert_is_array_or_matrix(v1)
        assert_is_array_or_matrix(v2)
        
        v1, v2 = self._to_compatible_matrix_types(v1, v2)
        # TODO: figure out where these asserts belong!!
        v1.assert_same_shape(v2)

        return self._sim(v1, v2) 
    
    def get_sims_to_matrix(self, vector, matrix_):
        
        assert_is_array_or_matrix(vector)
        assert_is_array_or_matrix(matrix_)
        
        vector, matrix_ = self._to_compatible_matrix_types(vector, matrix_)
        
        if vector.shape[1] != matrix_.shape[1] or vector.shape[0] != 1:
            raise ValueError("Inconsistent shapes %s %s" 
                             % (vector.shape, matrix_.shape))
        
        return self._sims_to_matrix(vector, matrix_)
    
    def _sims_to_matrix(self, vector, matrix_):
        
        result = np.zeros(shape=(matrix_.shape[0], 1))
        for i in range(matrix_.shape[0]):
            result[i] = self._sim(vector, matrix_[i,:])
        return type(matrix_)(result)    
        
        
    def _to_compatible_matrix_types(self, v1, v2):
    
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
        
        
    def get_name(self):
        return self._name
     
    def __str__(self):
        return self._name
    
    name = property(get_name)
    
