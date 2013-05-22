'''
Created on Oct 5, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import time
from warnings import warn
from composes.semantic_space.space import Space
from composes.matrix.dense_matrix import DenseMatrix
from composes.utils.gen_utils import assert_is_instance
from composes.utils.matrix_utils import resolve_type_conflict
from composes.utils.io_utils import create_parent_directories

import logging
from composes.utils import log_utils as log

logger = logging.getLogger(__name__)

class CompositionModel(object):
    """
    Parent class of the composition models.
    """
    
    _name = "no name"
    
    composed_id2column = None
    """
    List of strings, the column strings of the resulted composed space.
    """
    
    def __init__(self):
        """
        Constructor
        """
            
    def train(self, train_data, arg_space, phrase_space):
        """
        Trains a composition model and sets its learned parameters.
        
        Args:
            train_data: list of string tuples. Each tuple contains 3 
            string elements: (arg1, arg2, phrase).
            
            arg_space: argument space(s). Space object or a tuple of two 
            Space objects (e.g. my_space, or (my_space1, my_space2)). 
            If two spaces are provided, arg1 elements of train data are 
            interpreted in space1, and arg2 in space2.
        
            phrase space: phrase space, of type Space.
            
        Calls the specific training routine of the current composition
        model. Training tuples which contain strings not found in their 
        respective spaces are ignored.
        
        The id2column attribute of the resulted composed space is set to
        be equal to that of the phrase space given as an input.
        """
            
        start = time.time()
        
        arg1_space, arg2_space = self.extract_arg_spaces(arg_space)
        arg1_list, arg2_list, phrase_list = self.valid_data_to_lists(train_data,
                                                                     (arg1_space.row2id,
                                                                      arg2_space.row2id,
                                                                      phrase_space.row2id)
                                                                     )
        
        
        self.xxx(arg1_space, arg2_space, phrase_space,
                 arg1_list, arg2_list, phrase_list)
        
        self.composed_id2column = phrase_space.id2column
        
        log.print_composition_model_info(logger, self, 1, "\nTrained composition model:")
        log.print_info(logger, 2, "With total data points:%s" % len(arg1_list))
        log.print_matrix_info(logger, arg1_space.cooccurrence_matrix, 3, 
                              "Semantic space of argument 1:")
        log.print_matrix_info(logger, arg2_space.cooccurrence_matrix, 3, 
                              "Semantic space of argument 2:")
        log.print_matrix_info(logger, phrase_space.cooccurrence_matrix, 3, 
                              "Semantic space of phrases:")
        log.print_time_info(logger, time.time(), start, 2)
    

    def xxx(self, arg1_space, arg2_space, phrase_space, arg1_list, arg2_list, phrase_list):

        arg1_mat = arg1_space.get_rows(arg1_list)
        arg2_mat = arg2_space.get_rows(arg2_list)
        phrase_mat = phrase_space.get_rows(phrase_list)

        [arg1_mat, arg2_mat, phrase_mat] = resolve_type_conflict([arg1_mat, 
                                                                  arg2_mat,
                                                                  phrase_mat],
                                                                  DenseMatrix) 
        
        self._train(arg1_mat, arg2_mat, phrase_mat)

    def compose(self, data, arg_space):
        """
        Uses a composition model to compose elements.
        
        Args:
            data: data to be composed. List of tuples, each containing 3
            strings: (arg1, arg2, composed_phrase). arg1 and arg2 are the 
            elements to be composed and composed_phrase is the string associated
            to their composition.
            
            arg_space: argument space(s). Space object or a tuple of two 
            Space objects (e.g. my_space, or (my_space1, my_space2)). 
            If two spaces are provided, arg1 elements of data are 
            interpreted in space1, and arg2 in space2.
        
        Returns:
            composed space: a new object of type Space, containing the 
            phrases obtained through composition.
            
        """
        start = time.time()
         
        arg1_space, arg2_space = self.extract_arg_spaces(arg_space)
        arg1_list, arg2_list, phrase_list = self.valid_data_to_lists(data,
                                                                     (arg1_space.row2id,
                                                                      arg2_space.row2id,
                                                                      None))
                                                                     
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
        """
        TO BE MOVED TO A UTILS MODULE!
        """
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
        
 
    def valid_data_to_lists(self, data, (row2id1, row2id2, row2id3)):
        """
        TO BE MOVED TO A UTILS MODULE!
        """
        list1 = []
        list2 = []
        list3 = []
         
        j = 0
        for i in xrange(len(data)):
            sample = data[i]
            
            cond = True
            
            if not row2id1 is None:
                cond = cond and sample[0] in row2id1
            
            if not row2id2 is None:
                cond = cond and sample[1] in row2id2
            
            if not row2id3 is None:
                cond = cond and sample[2] in row2id3

            if cond:
                list1.append(sample[0]) 
                list2.append(sample[1])
                list3.append(sample[2])
                j += 1
    
        if i + 1 != j:
            warn("%d (out of %d) lines are ignored because one of the elements is not found in its semantic space"
                 % ((i + 1) - j, (i + 1)))
            
        if not list1:
            raise ValueError("No valid data found for training/composition!")
        
        return list1, list2, list3

    def export(self, filename):
        """
        Prints the parameters of the composition model to file.
        
        Args:
            filename: output filename, string
            
        Prints the parameters of the compositional model in an appropriate
        format, specific to each model.    
        """
        create_parent_directories(filename)
        self._export(filename)
    
    def get_name(self):
        return self._name
    
    name = property(get_name)    
    """
    String, name of the composition model.
    """
            
        
        