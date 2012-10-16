'''
Created on Sep 21, 2012

@author: georgianadinu
'''

from numpy import array
from numpy import prod
from warnings import warn
import time
from composes.utils.space_utils import list2dict
from composes.utils.space_utils import assert_dict_match_list
from composes.utils.space_utils import assert_shape_consistent
from composes.utils.space_utils import assert_is_instance 
from composes.utils.space_utils import add_items_to_dict
from composes.utils.matrix_utils2 import resolve_type_conflict
from composes.utils.matrix_utils2 import get_type_of_largest
from composes.matrix.matrix import Matrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.semantic_space.operation import FeatureSelectionOperation
from composes.semantic_space.operation import DimensionalityReductionOperation
from composes.similarity.similarity import Similarity
from composes.weighting.weighting import Weighting
from composes.dim_reduction.dimensionality_reduction import DimensionalityReduction
from composes.feature_selection.feature_selection import FeatureSelection
from composes.exception.illegal_state_error import IllegalOperationError
import logging
from composes.utils import log_utils as log
from composes.utils.space_utils import read_sparse_space_data
from composes.utils.space_utils import extract_indexing_structs
from composes.utils.space_utils import read_dense_space_data
from composes.utils.space_utils import read_words
from composes.utils.io_utils import create_parent_directories
from composes.utils.io_utils import print_list
from pyparsing import col

logger = logging.getLogger(__name__)


class Space(object):
    """
    This class implements semantic spaces.
    
    A semantic space described a list of targets (words, phrases, etc.)
    in terms of co-occurrence with contextual features. 
    
    A semantic space contains a matrix storing (some type of) co-occurrence
    strength values between targets and contextual features: by convention,
    targets are rows and features are columns.
    
    In addition to this co-occurrence matrix, the space stores structures
    that encode the mappings between the matrix row/column indices and the
    associated target/context-feature strings.
    
    A number of transformations can be applied to a semantic space, with the
    goal of improving the quality of the target representations. 
    Some transformations, such as weighings, only scale the values
    in the space matrix, while others, such as dimensionality
    reduction, or feature selection, alter the set of 
    contextual features.
    
    """

    def __init__(self, matrix_, id2row, id2column, row2id=None, column2id=None,
                 **kwargs):
        '''
        Constructor
        '''
        if row2id is None:
            row2id = list2dict(id2row)
        else:    
            assert_dict_match_list(row2id, id2row)
            
        if column2id is None:
            column2id = list2dict(id2column)
        else:
            assert_dict_match_list(column2id, id2column)
            
        assert_shape_consistent(matrix_, id2row, id2column, row2id, column2id)
        
        self._cooccurrence_matrix = matrix_
        self._row2id = row2id
        self._id2row = id2row
        self._column2id = column2id
        self._id2column = id2column
        if "operations" in kwargs:
            self._operations = kwargs["operations"]
        else:
            self._operations = []

        if "element_shape" in kwargs:
            elem_shape = kwargs["element_shape"]
            if prod(elem_shape) != self._cooccurrence_matrix.shape[1]:
                raise ValueError("Trying to assign invalid element shape:\
                                    element_shape: %s, matrix columns: %s" 
                                    % (str(elem_shape), 
                                       str(self._cooccurrence_matrix.shape[1])))
          
        # NOTE: watch out here, can cause bugs, if we change the dimension 
        # of a regular space and we do not create a new space         
            self._element_shape = kwargs["element_shape"]
        else:    
            self._element_shape = (self._cooccurrence_matrix.shape[1],)    

      
    def apply(self, transformation):
        
        start = time.time()
        #TODO , FeatureSelection, DimReduction ..
        assert_is_instance(transformation, (Weighting, DimensionalityReduction, 
                                            FeatureSelection))
        op = transformation.create_operation()
        new_matrix =  op.apply(self.cooccurrence_matrix)
        
        new_operations = list(self.operations)
        new_operations.append(op)

        id2row, row2id = list(self.id2row), self.row2id.copy() 
        
        if isinstance(op, DimensionalityReductionOperation):
            self.assert_1dim_element()
            id2column, column2id = [], {}
        elif isinstance(op, FeatureSelectionOperation):
            self.assert_1dim_element()
            op.original_columns = self.id2column
            id2column = list(array(op.original_columns)[op.selected_columns])
            column2id = list2dict(id2column)
        else:
            id2column, column2id = list(self.id2column), self.column2id.copy()

        log.print_transformation_info(logger, transformation, 1, 
                                      "\nApplied transformation:")
        log.print_matrix_info(logger, self.cooccurrence_matrix, 2, 
                              "Original semantic space:")
        log.print_matrix_info(logger, new_matrix, 2, "Resulted semantic space:")
        log.print_time_info(logger, time.time(), start, 2)
                        
        return Space(new_matrix, id2row, id2column,
                     row2id, column2id, operations = new_operations)
        
    def get_sim(self, word1, word2, similarity, space2=None):
        
        assert_is_instance(similarity, Similarity)
        v1 = self.get_row(word1)
        if space2 is None:
            v2 = self.get_row(word2)
        else:
            v2 = space2.get_row(word2)
        
        if v1 is None:
            warn("Row string %s not found, returning 0.0" % (word1))
            return 0.0

        if v2 is None:
            warn("Row string %s not found, returning 0.0" % (word2))
            return 0.0
        
        [v1, v2] = resolve_type_conflict([v1, v2], DenseMatrix)
        return similarity.get_sim(v1, v2)
      
    def get_neighbours(self, word, no_neighbours, similarity, 
                       neighbour_space=None):            
       
        start = time.time()
        assert_is_instance(similarity, Similarity)       
        vector = self.get_row(word)
        if vector is None:
            return []
        
        if neighbour_space is None:
            id2row = self.id2row
            sims_to_matrix = similarity.get_sims_to_matrix(vector, 
                                                          self.cooccurrence_matrix)
        else:
            mat_type = type(neighbour_space.cooccurrence_matrix)
            if not isinstance(vector, mat_type):
                vector = mat_type(vector)
            
            sims_to_matrix = similarity.get_sims_to_matrix(vector, 
                                         neighbour_space.cooccurrence_matrix)
            id2row = neighbour_space.id2row 
        
        sorted_perm = sims_to_matrix.sorted_permutation(sims_to_matrix.sum, 1)
        no_neighbours = min(no_neighbours, len(id2row))
        result = []
                
        for count in range(no_neighbours):
            i = sorted_perm[count]
            result.append((id2row[i], sims_to_matrix[i,0]))

        log.print_info(logger, 1, "\nGetting neighbours of:%s" % (word))
        log.print_name(logger, similarity, 1, "Similarity:")
        log.print_time_info(logger, time.time(), start, 2)
        return result    

    @classmethod
    def vstack(cls, space1, space2):
        if space1.cooccurrence_matrix.shape[1] != space2.cooccurrence_matrix.shape[1]:
            raise ValueError("Inconsistent shapes: %s, %s" 
                             % (space1.cooccurrence_matrix.shape[1], 
                                space2.cooccurrence_matrix.shape[1]))
        
        if space1.id2column != space2.id2column:
            raise ValueError("Identical columns required")
        
        new_row2id = add_items_to_dict(space1.row2id.copy(), space2.id2row)
        new_id2row = space1.id2row + space2.id2row
        
        matrix_type = get_type_of_largest([space1.cooccurrence_matrix,
                                           space2.cooccurrence_matrix])
        [new_mat1, new_mat2] = resolve_type_conflict([space1.cooccurrence_matrix, 
                                                      space2.cooccurrence_matrix],
                                                     matrix_type)
        
        new_mat = new_mat1.vstack(new_mat2)
        
        log.print_info(logger, 1, "\nVertical stack of two spaces")
        log.print_matrix_info(logger, space1.cooccurrence_matrix, 2, 
                              "Semantic space 1:")
        log.print_matrix_info(logger, space2.cooccurrence_matrix, 2, 
                              "Semantic space 2:")
        log.print_matrix_info(logger, new_mat, 2, "Resulted semantic space:")
        
        return Space(new_mat, new_id2row, list(space1.id2column), new_row2id, 
                     space1.column2id.copy(), operations=[])
        
    def to_dense(self):
        self.cooccurrence_matrix = DenseMatrix(self.cooccurrence_matrix)
        
    def to_sparse(self):
        self.cooccurrence_matrix = SparseMatrix(self.cooccurrence_matrix)
                
    def get_row(self, word):
        if not word in self.row2id:
            return None
        return self.cooccurrence_matrix[self.row2id[word],:]
    
    def get_rows(self, words):
        row_ids = []
        for word in words:
            if not word in self.row2id:
                raise ValueError("Word not found in space rows: %s" % word)
            else:
                row_ids.append(self.row2id[word])
        
        return self.cooccurrence_matrix[row_ids,:]
        
    def set_cooccurrence_matrix(self, matrix_):
        assert_is_instance(matrix_, Matrix)
        assert_shape_consistent(matrix_, self.row2id, self.id2row,
                                       self.column2id, self.id2column)
        self._cooccurrence_matrix = matrix_
        
    def get_cooccurrence_matrix(self):
        return self._cooccurrence_matrix
        
    cooccurrence_matrix = property(get_cooccurrence_matrix)
        
    def get_row2id(self):
        return self._row2id
            
    row2id = property(get_row2id)

    def get_id2row(self):
        return self._id2row
            
    id2row = property(get_id2row)

    def get_column2id(self):
        return self._column2id
            
    column2id = property(get_column2id)

    def get_id2column(self):
        return self._id2column
            
    id2column = property(get_id2column)
   
    def get_element_shape(self):
        return self._element_shape
            
    element_shape = property(get_element_shape)
     
    def get_operations(self):
        return self._operations
            
    operations = property(get_operations)
    
    def assert_1dim_element(self):
        if len(self.element_shape) > 1:
            raise IllegalOperationError("Operation not allowed on spaces with\
                                       element shape: %s" % self.element_shape)
   
    @classmethod
    def build(cls, **kwargs):
        # TODO: check arguments
        id2row = None
        id2column = None
        
        if "data" in kwargs:
            data_file = kwargs["data"]
        else:
            raise ValueError("Space data file needs to be specified")
            
        if "format" in kwargs:
            format_ = kwargs["format"]
            if not format_ in ["dm","sm"]:
                raise ValueError("Unrecognized format: %s" % format_)
        else:
            raise ValueError("Format of input files needs to be specified")
        
        if "rows" in kwargs:
            [id2row], [row2id] = extract_indexing_structs(kwargs["rows"], [0])
        if "cols" in kwargs:
            [id2column], [column2id] = extract_indexing_structs(kwargs["cols"], [0])
        
        if format_ == "sm":
            if id2row is None and id2column is None:
                ([id2row, id2column],
                 [row2id, column2id]) = extract_indexing_structs(data_file, [0, 1])
            if id2row is None:
                [id2row], [row2id] = extract_indexing_structs(data_file, [0])
            if id2column is None:
                [id2column], [column2id] = extract_indexing_structs(data_file, [1])
                
            mat = read_sparse_space_data(data_file, row2id, column2id)
        else:
            if id2row is None:
                [id2row],[row2id] = extract_indexing_structs(data_file, [0])
            if id2column is None:

                id2column, column2id = [], {}
             
            mat = read_dense_space_data(data_file, row2id)
                
        if id2column and len(id2column) != mat.shape[1]:
            raise ValueError("Columns provided inconsistent with shape of input matrix!")
            
        return Space(mat, id2row, id2column, row2id, column2id)
    
    def export(self, file_prefix, **kwargs):
        create_parent_directories(file_prefix)
        format_ = "dm"
        if "format" in kwargs:
            format_ = kwargs["format"]
            if format_ != "dm" and format_ != "sm":
                raise ValueError("Unrecognized format: %s" %format_)
            elif format_ == "dm":
                self._dense_format_export(file_prefix)
            else:
                self._sparse_format_export(file_prefix)
        self._export_row_column(file_prefix)
        
    def _export_row_column(self, file_prefix):
        row_file = "%s.%s" %(file_prefix, "row")
        column_file = "%s.%s" %(file_prefix, "col")
        
        if self.column2id != []:
            print_list(self.id2column, column_file)
        print_list(self.id2row, row_file)

    def _sparse_format_export(self, file_prefix):
        matrix_file = "%s.%s" %(file_prefix, "sm")
        with open(matrix_file, 'w') as f: 
            if isinstance(self.cooccurrence_matrix,SparseMatrix):
                mat = self.cooccurrence_matrix.mat

                data = mat.data
                row_indices = mat.indptr
                col_indices = mat.indices
                
                row_index = 0;
                next_row = row_indices[1]
                row = self.id2row[0]  
                for i in xrange(len(data)):
                    while i == next_row:
                        row_index +=1
                        next_row = row_indices[row_index + 1]
                        row = self.id2row[row_index]  
                    col = self.id2column[col_indices[i]]
                    f.write("%s\t%s\t%f\n" %(row,col,data[i]))
        
    def _dense_format_export(self, file_prefix):
        matrix_file = "%s.%s" %(file_prefix, "dm")
        with open(matrix_file, 'w') as f: 
            for i, row in enumerate(self.id2row):
                v = DenseMatrix(self.cooccurrence_matrix[i]).mat.flat
                f.write("\t".join([row] + [repr(v[j]) for j in range(len(v))]))
