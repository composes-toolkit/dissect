'''
Created on Oct 11, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

import time
import logging

import numpy as np

from composition_model import CompositionModel
from composes.semantic_space.space import Space
from composes.utils.gen_utils import get_partitions
from composes.utils.gen_utils import assert_valid_kwargs
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.utils.regression_learner import RegressionLearner
from composes.utils.matrix_utils import resolve_type_conflict
from composes.utils.matrix_utils import get_type_of_largest
from composes.utils.matrix_utils import padd_matrix
from composes.utils.num_utils import is_integer
from composes.utils.gen_utils import assert_is_instance
from composes.exception.illegal_state_error import IllegalStateError
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
    _MIN_SAMPLES = 1

    def __init__(self, **kwargs):
        """
        Constructor.
        
        Args:
            function_space= : function space parameter, containing
            the lexical functions, of type Space. Optional, can be set through
            training.
            
            intercept= : True/False, True if the function space has intercept.
            Optional, default False. When training is used, intercept is set 
            to the intercept value of the regression learner used.
        
            learner= : regression method of type RegressionLearner. Optional,
            default LstsqRegressionLearner.
            
        """
        assert_valid_kwargs(kwargs, ["function_space", "intercept", "learner"])

        self._regression_learner = LstsqRegressionLearner()
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
        """
        Trains a lexical function composition model to learn a function
        space and sets the function_space parameter. 
                
        Args:
            train_data: list of string tuples. Each tuple contains 3 
            string elements: (function_word, arg, phrase).
            
            arg_space: argument space, of type Space. arg elements of 
            train data are interpreted in this space.
        
            phrase space: phrase space, of type Space. phrase elements of 
            the train data are interpreted in this space.
            
        Training tuples which contain strings not found in their 
        respective spaces are ignored. Function words containing less than
        _MIN_SAMPLES training instances are ignored. For example, if
        _MIN_SAMPLES=2 and function word "red" occurs in only one phrase, "red"
        is ignored.
        
        The id2column attribute of the resulted composed space is set to
        be equal to that of the phrase space given as an input.
        """

        start = time.time()

        self._has_intercept = self._regression_learner.has_intercept()

        if not isinstance(arg_space, Space):
            raise ValueError("expected one input spaces!")

        result_mats = []

        train_data = sorted(train_data, key=lambda tup: tup[0])
        function_word_list, arg_list, phrase_list = self.valid_data_to_lists(train_data,
                                                                             (None,
                                                                              arg_space.row2id,
                                                                              phrase_space.row2id))
        #partitions the sorted input data
        keys, key_ranges = get_partitions(function_word_list, self._MIN_SAMPLES)

        if not keys:
            raise ValueError("No valid training data found!")

        assert (len(arg_space.element_shape) == 1)

        if self._has_intercept:
            new_element_shape = phrase_space.element_shape + (arg_space.element_shape[0] + 1,)
        else:
            new_element_shape = phrase_space.element_shape + (arg_space.element_shape[0],)

        for i in xrange(len(key_ranges)):
            idx_beg, idx_end = key_ranges[i]

            print ("Training lexical function...%s with %d samples"
                   % (keys[i], idx_end - idx_beg))

            arg_mat = arg_space.get_rows(arg_list[idx_beg:idx_end])
            phrase_mat = phrase_space.get_rows(phrase_list[idx_beg:idx_end])

            #convert them to the same type
            matrix_type = get_type_of_largest([arg_mat, phrase_mat])
            [arg_mat, phrase_mat] = resolve_type_conflict([arg_mat, phrase_mat],
                                                          matrix_type)

            result_mat = self._regression_learner.train(arg_mat, phrase_mat).transpose()

            result_mat.reshape((1, np.prod(new_element_shape)))

            result_mats.append(result_mat)

        new_space_mat = arg_mat.nary_vstack(result_mats)

        self.composed_id2column = phrase_space.id2column

        self._function_space = Space(new_space_mat, keys, [],
                                     element_shape=new_element_shape)

        log.print_composition_model_info(logger, self, 1, "\nTrained composition model:")
        log.print_info(logger, 3, "Trained: %s lexical functions" % len(keys))
        log.print_info(logger, 3, "With total data points:%s" % len(function_word_list))
        log.print_matrix_info(logger, arg_space.cooccurrence_matrix, 3,
                              "Semantic space of arguments:")
        log.print_info(logger, 3, "Shape of lexical functions learned:%s"
                                  % (new_element_shape,))
        log.print_matrix_info(logger, new_space_mat, 3,
                              "Semantic space of lexical functions:")
        log.print_time_info(logger, time.time(), start, 2)

    def compose(self, data, arg_space):
        """
        Uses a lexical function composition model to compose elements.
        
        Args:
            data: data to be composed. List of tuples, each containing 3
            strings: (function_word, arg, composed_phrase). function_word and 
            arg are the elements to be composed and composed_phrase is the 
            string associated to their composition. function_word elements
            are interpreted in self.function_space. 
            
            arg_space: argument space, of type Space. arg elements of data are 
            interpreted in this space. 
        
        Returns:
            composed space: a new object of type Space, containing the 
            phrases obtained through composition.
            
        """
        start = time.time()

        assert_is_instance(arg_space, Space)
        arg1_list, arg2_list, phrase_list = self.valid_data_to_lists(data,
                                                                     (self._function_space.row2id,
                                                                      arg_space.row2id,
                                                                      None))

        composed_vec_list = []
        for i in xrange(len(arg1_list)):
            arg1_vec = self._function_space.get_row(arg1_list[i])
            arg2_vec = arg_space.get_row(arg2_list[i])

            matrix_type = get_type_of_largest([arg1_vec, arg2_vec])
            [arg1_vec, arg2_vec] = resolve_type_conflict([arg1_vec, arg2_vec],
                                                         matrix_type)

            composed_ph_vec = self._compose(arg1_vec, arg2_vec,
                                            self._function_space.element_shape)

            composed_vec_list.append(composed_ph_vec)

        result_element_shape = self._function_space.element_shape[0:-1]
        composed_ph_mat = composed_ph_vec.nary_vstack(composed_vec_list)

        log.print_name(logger, self, 1, "\nComposed with composition model:")
        log.print_info(logger, 3, "Composed total data points:%s" % len(arg1_list))
        log.print_info(logger, 3, "Functional shape of the resulted (composed) elements:%s"
                                  % (result_element_shape,))
        log.print_matrix_info(logger, composed_ph_mat, 4,
                              "Resulted (composed) semantic space:")
        log.print_time_info(logger, time.time(), start, 2)

        return Space(composed_ph_mat, phrase_list, self.composed_id2column,
                     element_shape=result_element_shape)


    def _compose(self, function_arg_vec, arg_vec, function_arg_element_shape):

        new_shape = (np.prod(function_arg_element_shape[0:-1]),
                     function_arg_element_shape[-1])

        function_arg_vec.reshape(new_shape)

        if self._has_intercept:
            comp_el = function_arg_vec * padd_matrix(arg_vec.transpose(), 0)
        else:
            comp_el = function_arg_vec * arg_vec.transpose()

        return comp_el.transpose()

    @classmethod
    def _assert_space_match(cls, arg1_space, arg2_space, phrase_space=None):
        pass

    def set_regression_learner(self, regression_learner):
        assert_is_instance(regression_learner, RegressionLearner)
        self._regression_learner = regression_learner

    def get_regression_learner(self):
        return self._regression_learner

    regression_learner = property(get_regression_learner, set_regression_learner)
    """
    Regression method to be used in training, of type RegressionLearner.
    Default is RidgeRegressionLearner(param=1).
    """

    def get_function_space(self):
        return self._function_space

    function_space = property(get_function_space)
    """
    Function space parameter, containing the lexical functions, of type Space. 
    Can be set through training or through initialization, default None.
    """

    def get_has_intercept(self):
        return self._has_intercept

    has_intercept = property(get_has_intercept)
    """
    Has intercept parameter, boolean. If True, then the function_space is 
    assumed to contain intercept. Can be set through training or through 
    initialization, default is assumed to be False.
    """

    def set_min_samples(self, min_samples):
        if not is_integer(min_samples):
            raise ValueError("expected %s min_samples value, received %s"
                             % ("integer", type(min_samples)))
        self._MIN_SAMPLES = min_samples

    def get_min_samples(self):
        return self._MIN_SAMPLES

    MIN_SAMPLES = property(get_min_samples, set_min_samples)
    """
    Minimal number of samples for each training instance. Default 3.
    """

    def _export(self, filename):
        if self._function_space is None:
            raise IllegalStateError("cannot export an untrained LexicalFunction model.")
        self._function_space.export(filename, format="dm")
            
        
            