'''
Created on Oct 11, 2012

@author: georgianadinu
'''

import numpy as np
from composition_model import CompositionModel
from composes.semantic_space.space import Space
from collections import Counter
from composes.utils.regression_learner import RidgeRegressionLearner
from composes.utils.matrix_utils2 import resolve_type_conflict

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
        
        if "learner" in kwargs:
            self._regression_learner = kwargs["learner"] 
            self._has_intercept = self._regression_learner.has_intercept()
        
    def train(self):
        raise NotImplementedError("Cannot train lexical function model!\
                                   Use learn_function_space instead!")
        
    def learn_function_space(self, train_data, arg_space, phrase_space):
 
 
        if not isinstance(arg_space, Space):
            raise ValueError("expected one input spaces!")  
               
        result_mats = []
               
        train_data = sorted(train_data, key=lambda tup: tup[0])
        function_word_list, arg_list, phrase_list = self.data_to_lists(train_data)

        freqs = Counter(function_word_list)
        keys = freqs.keys()
        keys.sort()
        idx = 0

        for function_word in keys:
            count = freqs[function_word]

            arg_mat = arg_space.get_rows(arg_list[idx:idx+count]) 
            phrase_mat = phrase_space.get_rows(phrase_list[idx:idx+count])
            idx += count

            result_mat = self._regression_learner.train(arg_mat, phrase_mat)
            result_mats.append(result_mat.transpose())

        new_space_mat = arg_mat.nary_vstack(result_mats)
        new_element_shape = phrase_space.element_shape + arg_space.element_shape    
        new_space_mat.reshape((len(keys), np.prod(new_element_shape)))

        self.composed_id2column = phrase_space.id2column
            
        return Space(new_space_mat, keys, [], element_shape=new_element_shape)
        
        
    def compose(self, data, arg_space):
        
        if not isinstance(arg_space, tuple):
            raise ValueError("expected two input spaces!")  
        
        arg1_space, arg2_space = self.extract_arg_spaces(arg_space)
        arg1_list, arg2_list, phrase_list = self.data_to_lists(data)

        arg1_mat = arg1_space.get_rows(arg1_list)
        arg2_mat = arg2_space.get_rows(arg2_list)
        
        composed_ph_mat, composed_elem_shape = self._compose(arg1_mat,
                                                             arg2_mat,
                                                             arg1_space.element_shape)

        return Space(composed_ph_mat, phrase_list, self.composed_id2column, 
                     element_shape = composed_elem_shape)
    
        
    def _compose(self, arg1_mat, arg2_mat, arg1_element_shape):
        
        result = []
        new_shape = (np.prod(arg1_element_shape[0:-1]), arg1_element_shape[-1])
                 
        for i in range(arg2_mat.shape[0]):
            
            arg1_mat_row = arg1_mat[i]
            arg1_mat_row.reshape(new_shape)
            comp_el = arg1_mat_row * arg2_mat[i].transpose()
            result.append(comp_el.transpose())
            
        result = arg1_mat.nary_vstack(result)
        result_element_shape = arg1_element_shape[0:-1]
        
        return result, result_element_shape    
            
    @classmethod
    def _assert_space_match(cls, arg1_space, arg2_space, phrase_space=None):
        pass
            
        
        
        
        
            