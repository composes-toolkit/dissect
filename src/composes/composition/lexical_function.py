'''
Created on Oct 11, 2012

@author: georgianadinu
'''

import numpy as np
from composition_model import CompositionModel
from composes.semantic_space.space import Space
from composes.utils.space_utils import get_partitions
from composes.utils.regression_learner import RidgeRegressionLearner
from composes.utils.matrix_utils2 import resolve_type_conflict
from composes.utils.matrix_utils2 import get_type_of_largest
from composes.utils.matrix_utils2 import padd_matrix
from composes.utils.space_utils import assert_is_instance

class LexicalFunction(CompositionModel):
    '''
    classdocs
    '''
    _name = "lexical_function"

    def __init__(self, **kwargs):
        '''
        Constructor
        '''
        self._regression_learner = RidgeRegressionLearner(intercept=False)
        self.composed_id2column = []
        self._function_space = None
        self._has_intercept = False
        
        if "function_space" in kwargs:
            space = kwargs["function_space"]
            if not isinstance(space, Space):
                raise TypeError("expected Space-type argument, received:" 
                                 % type(space))
            self._function_space = kwargs["function_space"]

        if "intercept" in kwargs:
            has_intercept = kwargs["intercept"]
            if not isinstance(has_intercept, bool):
                raise TypeError("expected bool-type argument, received:" 
                                 % type(has_intercept))
            self._has_intercept = has_intercept
            
        if "learner" in kwargs:
            if "function_space" in kwargs:
                raise ValueError("cannot instantiate with both learner and function_space!")
             
            self._regression_learner = kwargs["learner"] 
        
    def train(self, train_data, arg_space, phrase_space):
 
        self._has_intercept = self._regression_learner.has_intercept()
        if not isinstance(arg_space, Space):
            raise ValueError("expected one input spaces!")  
               
        result_mats = []
               
        train_data = sorted(train_data, key=lambda tup: tup[0])
        function_word_list, arg_list, phrase_list = self.data_to_lists(train_data)

        #this is the list we wanted
        keys, new_key_indices = get_partitions(function_word_list)
        
        for i in range(0, len(new_key_indices) - 1):

            idx_beg, idx_end = new_key_indices[i], new_key_indices[i + 1]            

            arg_mat = arg_space.get_rows(arg_list[idx_beg:idx_end]) 
            phrase_mat = phrase_space.get_rows(phrase_list[idx_beg:idx_end])
 
            #convert them to the same type
            matrix_type = get_type_of_largest([arg_mat, phrase_mat])
            [arg_mat, phrase_mat] = resolve_type_conflict([arg_mat, phrase_mat],
                                                          matrix_type)
 
            result_mat = self._regression_learner.train(arg_mat, phrase_mat)
            result_mats.append(result_mat.transpose())

        new_space_mat = arg_mat.nary_vstack(result_mats)
        assert(len(arg_space.element_shape) == 1)
        
        if self._has_intercept:
            new_element_shape = phrase_space.element_shape + (arg_space.element_shape[0] + 1,)
        else:
            new_element_shape = phrase_space.element_shape + (arg_space.element_shape[0],)
                    
        new_space_mat.reshape((len(keys), np.prod(new_element_shape)))

        self.composed_id2column = phrase_space.id2column
            
        self._function_space = Space(new_space_mat, keys, [], 
                                    element_shape=new_element_shape)
        
        
    def compose(self, data, arg_space):
        
        assert_is_instance(arg_space, Space)
        arg1_list, arg2_list, phrase_list = self.data_to_lists(data)

        arg1_mat = self._function_space.get_rows(arg1_list)
        arg2_mat = arg_space.get_rows(arg2_list)
        
        matrix_type = get_type_of_largest([arg1_mat, arg2_mat])
        [arg1_mat, arg2_mat] = resolve_type_conflict([arg1_mat, arg2_mat],
                                                          matrix_type)
            
        composed_ph_mat, composed_elem_shape = self._compose(arg1_mat,
                                                             arg2_mat,
                                                             self._function_space.element_shape)

        return Space(composed_ph_mat, phrase_list, self.composed_id2column, 
                     element_shape = composed_elem_shape)
    
        
    def _compose(self, function_arg_mat, arg_mat, function_arg_element_shape):
        
        result = []
        new_shape = (np.prod(function_arg_element_shape[0:-1]), 
                            function_arg_element_shape[-1])
                 
        for i in range(arg_mat.shape[0]):
            
            function_arg_mat_row = function_arg_mat[i]
            function_arg_mat_row.reshape(new_shape)
            if self._has_intercept:
                comp_el = function_arg_mat_row * padd_matrix(arg_mat[i].transpose(), 0)
            else:    
                comp_el = function_arg_mat_row * arg_mat[i].transpose()
            result.append(comp_el.transpose())
            
        result = function_arg_mat.nary_vstack(result)
        result_element_shape = function_arg_element_shape[0:-1]
        
        return result, result_element_shape    
            
    @classmethod
    def _assert_space_match(cls, arg1_space, arg2_space, phrase_space=None):
        pass
    
    def get_function_space(self):
        return self._function_space
    
    function_space = property(get_function_space)
    
    
            
        
        
        
        
            