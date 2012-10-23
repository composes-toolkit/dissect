'''
Created on Oct 11, 2012

@author: georgianadinu
'''

import numpy as np
import time
from warnings import warn
from composition_model import CompositionModel
from composes.semantic_space.space import Space
from composes.utils.space_utils import get_partitions
from composes.utils.regression_learner import RidgeRegressionLearner
from composes.utils.regression_learner import RegressionLearner
from composes.utils.matrix_utils import resolve_type_conflict
from composes.utils.matrix_utils import get_type_of_largest
from composes.utils.matrix_utils import padd_matrix
from composes.utils.space_utils import assert_is_instance
from composes.exception.illegal_state_error import IllegalStateError

import logging
from composes.utils import log_utils as log

logger = logging.getLogger(__name__)

class LexicalFunction(CompositionModel):
    """
    Implements the lexical function compositional model.
    
        :math:`\\vec{p} = U \\vec{v}`
     
    where :math:`\\vec{p}` is the vector of the composed phrase,
    :math:`U` is the matrix representation of the first component (the lexical function)
    and :math:`\\vec{v}` is the vector representation of the second component
          
    """ 
         
    _name = "lexical_function"
    _MIN_SAMPLES = 0

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
        
        start = time.time()
 
        # TODO: VERY IMPORTANT lower limit on number of instances per function here!
        self._has_intercept = self._regression_learner.has_intercept()
        if not isinstance(arg_space, Space):
            raise ValueError("expected one input spaces!")  
               
        result_mats = []
               
        train_data = sorted(train_data, key=lambda tup: tup[0])
        function_word_list, arg_list, phrase_list = self.lf_valid_data_to_lists(train_data,
                                                                                arg_space.id2row,
                                                                                phrase_space.id2row)

        #this is the list we wanted
        keys, new_key_indices = get_partitions(function_word_list)
        
        for i in range(0, len(new_key_indices) - 1):

            idx_beg, idx_end = new_key_indices[i], new_key_indices[i + 1]
             
            if idx_end - idx_beg < self._MIN_SAMPLES:
                warn("Lexical function training. Ignoring:%s. No. samples:%d" 
                     % (new_key_indices[i], idx_end - idx_beg))
                keys.remove(keys[i])
                continue
                            
            arg_mat = arg_space.get_rows(arg_list[idx_beg:idx_end]) 
            phrase_mat = phrase_space.get_rows(phrase_list[idx_beg:idx_end])
 
            #convert them to the same type
            matrix_type = get_type_of_largest([arg_mat, phrase_mat])
            [arg_mat, phrase_mat] = resolve_type_conflict([arg_mat, phrase_mat],
                                                          matrix_type)

            result_mat = self._regression_learner.train(arg_mat, phrase_mat)
            result_mats.append(result_mat.transpose())

        #TODO: HERE AND IN OTHER PLACES; WHAT IF THERE NOTHING LEFT TO STACK??
        #NONe of the training data was valid!!
        if not result_mats:
            raise ValueError("No valid train data found!")
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
        
        log.print_composition_model_info(logger, self, 1, "\nTrained composition model:")
        log.print_info(logger, 3, "Trained: %s lexical functions" % len(keys))
        log.print_info(logger, 3, "With total data points:%s" % len(train_data))
        log.print_matrix_info(logger, arg_space.cooccurrence_matrix, 3, 
                              "Semantic space of arguments:")
        log.print_info(logger, 3, "Shape of lexical functions learned:%s" 
                       % (new_element_shape,))
        log.print_matrix_info(logger, new_space_mat, 3, 
                              "Semantic space of lexical functions:")
        
        log.print_time_info(logger, time.time(), start, 2)
        
    def compose(self, data, arg_space):
        
        start = time.time()
        
        assert_is_instance(arg_space, Space)
        arg1_list, arg2_list, phrase_list = self.valid_data_to_lists(data,
                                                                     self._function_space.id2row,
                                                                     arg_space.id2row)

        arg1_mat = self._function_space.get_rows(arg1_list)
        arg2_mat = arg_space.get_rows(arg2_list)
        
        matrix_type = get_type_of_largest([arg1_mat, arg2_mat])
        [arg1_mat, arg2_mat] = resolve_type_conflict([arg1_mat, arg2_mat],
                                                          matrix_type)
            
        composed_ph_mat, composed_elem_shape = self._compose(arg1_mat,
                                                             arg2_mat,
                                                             self._function_space.element_shape)

        log.print_name(logger, self, 1, "\nComposed with composition model:")
        log.print_info(logger, 3, "Composed total data points:%s" % arg1_mat.shape[0])
        log.print_info(logger, 3, "Functional shape of the resulted (composed) elements:%s" 
                       % (composed_elem_shape,))
        log.print_matrix_info(logger, composed_ph_mat, 4, 
                              "Resulted (composed) semantic space:")
        log.print_time_info(logger, time.time(), start, 2)
        
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
 
    def set_regression_learner(self, regression_learner):
        assert_is_instance(regression_learner, RegressionLearner)
        self._regression_learner = regression_learner
        
    def get_regression_learner(self):
        return self._regression_learner
    regression_learner = property(get_regression_learner, set_regression_learner)  
       
    def get_function_space(self):
        return self._function_space
    
    function_space = property(get_function_space)
    
    def _export(self, filename):
        if self._function_space is None:
            raise IllegalStateError("cannot export an untrained LexicalFunction model.")
        self._function_space.export(filename, format="dm")
            
        
        
        
        
            