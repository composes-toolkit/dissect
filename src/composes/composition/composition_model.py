'''
Created on Oct 5, 2012

@author: georgianadinu
'''
import time
from composes.semantic_space.space import Space
from composes.matrix.dense_matrix import DenseMatrix
from composes.utils.space_utils import assert_is_instance
from composes.utils.matrix_utils2 import resolve_type_conflict
from composes.utils.io_utils import create_parent_directories

import logging
from composes.utils import log_utils as log

logger = logging.getLogger(__name__)

class CompositionModel(object):
    '''
    classdocs
    '''
    _name = "no name"
    composed_id2column = None

    def __init__(self):
        '''
        Constructor
        '''
            
    def train(self, train_data, arg_space, phrase_space):    
        start = time.time()
        
        arg1_space, arg2_space = self.extract_arg_spaces(arg_space)
        arg1_list, arg2_list, phrase_list = self.data_to_lists(train_data)

        arg1_mat = arg1_space.get_rows(arg1_list)
        arg2_mat = arg2_space.get_rows(arg2_list)
        phrase_mat = phrase_space.get_rows(phrase_list)

        [arg1_mat, arg2_mat, phrase_mat] = resolve_type_conflict([arg1_mat, 
                                                                  arg2_mat,
                                                                  phrase_mat],
                                                                  DenseMatrix) 
        self._train(arg1_mat, arg2_mat, phrase_mat)
        self.composed_id2column = phrase_space.id2column
        
        log.print_composition_model_info(logger, self, 1, "\nTrained composition model:")
        log.print_info(logger, 2, "With total data points:%s" % arg1_mat.shape[0])
        log.print_matrix_info(logger, arg1_space.cooccurrence_matrix, 3, 
                              "Semantic space of argument 1:")
        log.print_matrix_info(logger, arg2_space.cooccurrence_matrix, 3, 
                              "Semantic space of argument 2:")
        log.print_matrix_info(logger, phrase_space.cooccurrence_matrix, 3, 
                              "Semantic space of phrases:")
        log.print_time_info(logger, time.time(), start, 2)
    
    
    def compose(self, data, arg_space):
        start = time.time()
         
        arg1_space, arg2_space = self.extract_arg_spaces(arg_space)
        arg1_list, arg2_list, phrase_list = self.data_to_lists(data)

        arg1_mat = arg1_space.get_rows(arg1_list)
        arg2_mat = arg2_space.get_rows(arg2_list)
        
        [arg1_mat, arg2_mat] = resolve_type_conflict([arg1_mat, arg2_mat], DenseMatrix) 
        
        composed_phrase_mat = self._compose(arg1_mat, arg2_mat)
        if self.composed_id2column is None:
            self.composed_id2column = self._build_id2column(arg1_space, arg2_space)

        log.print_name(logger, self, 1, "\nComposed with composition model:")
        log.print_info(logger, 3, "Composed total data points:%s" % arg1_mat.shape[0])
        log.print_matrix_info(logger, composed_phrase_mat, 4, 
                              "Resulted (composed) semantic space::")
        log.print_time_info(logger, time.time(), start, 2)
                
        return Space(composed_phrase_mat, phrase_list, self.composed_id2column)
    
    @classmethod
    def extract_arg_spaces(cls, arg_space):
    
        if not isinstance(arg_space, tuple):
            arg1_space = arg_space
            arg2_space = arg_space
        else:
            if len(arg_space) != 2:
                raise ValueError("expected two spaces, received %d-ary tuple " 
                                 % len(arg_space))
            arg1_space, arg2_space = arg_space
      
        assert_is_instance(arg1_space, Space)
        assert_is_instance(arg2_space, Space)
         
        cls._assert_space_match(arg1_space, arg2_space)
         
        return arg1_space, arg2_space 
    
    @classmethod
    def _assert_space_match(cls, arg1_space, arg2_space, phrase_space=None):
        
        if arg1_space.id2column != arg2_space.id2column:
            raise ValueError("Argument spaces do not have identical columns!") 
        
        if not phrase_space is None:
            if arg1_space.id2column != phrase_space.id2column:
                raise ValueError("Argument and phrase space do not have identical columns!")
    
    def _build_id2column(self, arg1_space, arg2_space):
        return arg1_space.id2column
        
        
    def data_to_lists(self, data):
                
        arg1_list = self.extract_list_from_tuples(data, 0)
        arg2_list = self.extract_list_from_tuples(data, 1)
        phrase_list = self.extract_list_from_tuples(data, 2)
        
        return arg1_list, arg2_list, phrase_list 
                
    def extract_list_from_tuples(self, tuples, position):        
        return [tuples[i][position] for i in xrange(len(tuples))]
    
    def export(self, filename):
        create_parent_directories(filename)
        self._export(filename)
    
    def get_name(self):
        return self._name
    name = property(get_name)    
        
        
        