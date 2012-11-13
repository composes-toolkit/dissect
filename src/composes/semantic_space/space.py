'''
Created on Sep 21, 2012

@author: georgianadinu
'''

import time
import logging
from numpy import array
from numpy import prod
from composes.utils.space_utils import list2dict
from composes.utils.space_utils import assert_dict_match_list
from composes.utils.space_utils import assert_shape_consistent
from composes.utils.space_utils import assert_is_instance 
from composes.utils.space_utils import add_items_to_dict
from composes.utils.matrix_utils import resolve_type_conflict
from composes.utils.matrix_utils import get_type_of_largest
from composes.matrix.matrix import Matrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.semantic_space.operation import FeatureSelectionOperation
from composes.semantic_space.operation import DimensionalityReductionOperation
from composes.similarity.similarity import Similarity
from composes.transformation.scaling.scaling import Scaling
from composes.transformation.dim_reduction.dimensionality_reduction import DimensionalityReduction
from composes.transformation.feature_selection.feature_selection import FeatureSelection
from composes.exception.illegal_state_error import IllegalOperationError
from composes.utils import log_utils as log
from composes.utils.space_utils import read_sparse_space_data
from composes.utils.io_utils import extract_indexing_structs
from composes.utils.space_utils import read_dense_space_data
from composes.utils.io_utils import create_parent_directories
from composes.utils.io_utils import print_list
from composes.utils.io_utils import print_cooc_mat_dense_format
from composes.utils.io_utils import print_cooc_mat_sparse_format


logger = logging.getLogger(__name__)


class Space(object):
    """
    This class implements semantic spaces.
    
    A semantic space describes a list of targets (words, phrases, etc.)
    in terms of co-occurrence with contextual features. 
    
    It contains a matrix storing (some type of) co-occurrence
    strength values between targets and contextual features: by convention,
    targets are rows and features are columns. The space also stores structures
    that encode the mappings between the matrix row/column indices and the
    associated target/context-feature strings.
    
    Transformations which rescale the matrix elements can be applied 
    to a semantic space. A semantic also space allows for similarity 
    computations between row elements of the space. 
    
    """

    def __init__(self, matrix_, id2row, id2column, row2id=None, column2id=None,
                 **kwargs):
        """
        Constructor.
        
        Args:
            matrix_: Matrix type, the data matrix of the space
            id2row: list, the row elements
            id2column: list, the column elements
            row2id: dictionary, maps row strings to ids. Optional, built from 
                id2row by default.
            column2id: dictionary, maps col strings to ids. Optional, built
                from id2column by default
            operations: list of operations already performed on the input
                matrix, Optional, by default set to empty.
            element_shape: tuple of int, the shape on row elements. Optional, 
                by default row elements are one-dimensional and element_shape is
                (no_cols, ). Used in 3D composition.
             
         Returns:
             A semantic space (type Space)
             
         Raises:
             TypeError: if matrix_ is not of the correct type
             ValueError: if element shape is not consistent with 
                         the size of matrix rows
                         if the matrix and the provided row and column 
                         indexing structures are not of consistent shapes.
                 
        """
        assert_is_instance(matrix_, Matrix)
        
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
        """
        Applies a transformation on the current space.
        
        All transformations affect the data matrix. If the transformation 
        reduces the dimensionality of the space, the column indexing
        structures are also updated. The operation applied is appended
        to the list of operations that the space holds. 
        
        Args:
            transformation: of type Scaling, DimensionalityReduction or 
              FeatureSelection  

        Returns:
            A new space on which the transformation has been applied.
            
        """
        start = time.time()
        #TODO , FeatureSelection, DimReduction ..
        assert_is_instance(transformation, (Scaling, DimensionalityReduction, 
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
            
            if op.original_columns: 
                id2column = list(array(op.original_columns)[op.selected_columns])
                column2id = list2dict(id2column)
            else:
                id2column, column2id = [],{}
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
        """
        Computes the similarity between two targets in the semantic 
        space.

        If one of the two targets to be compared is not found, it returns 0..
        
        Args:
            word1: string
            word2: string
            similarity: of type Similarity, the similarity measure to be used
            space2: Space type, Optional. If provided, word2 is interpreted in 
                this space, rather than the current space. Default, both words
                are interpreted in the current space.
        Returns:
            scalar, similarity score
            
        """
        
        assert_is_instance(similarity, Similarity)
        
        try:
            v1 = self.get_row(word1)
        except KeyError:
            print "Row string %s not found, returning 0.0" % (word1)
            return 0.0
        try:
            if space2 is None:
                v2 = self.get_row(word2)
            else:
                v2 = space2.get_row(word2)
        except KeyError:
            print "Row string %s not found, returning 0.0" % (word2)
            return 0.0
                
        [v1, v2] = resolve_type_conflict([v1, v2], DenseMatrix)
        return similarity.get_sim(v1, v2)

    def get_sims(self, word_pair_list, similarity, space2=None):
        """
        Computes the similarity between two LIST of targets in the semantic 
        space.

        If one of the two targets to be compared is not found, it returns 0..
        
        Args:
            word_pair_list: list of (string, string) tuples. Words to be compared.
            similarity: of type Similarity, the similarity measure to be used
            space2: Space type, Optional. If provided, the second word of the word pairs
                is interpreted in this space, rather than the current space. 
                Default, both words are interpreted in the current space.
        Returns:
            list, list of similarity scores
            
        """
        sims = []
        
        for word1, word2 in word_pair_list:
            sims.append(self.get_sim(word1, word2, similarity, space2))
        
        return sims
          
    def get_neighbours(self, word, no_neighbours, similarity, 
                       space2=None):            
        """
        Computes the neighbours of a word in the semantic space.

        Args:
            word: string, target word
            no_neighbours: int, the number of neighbours desired
            similarity: of type Similarity, the similarity measure to be used
            space2: Space type, Optional. If provided, the neighbours are 
                retrieved from this space, rather than the current space. 
                Default, neighbours are retrieved from the current space.
                
        Returns:
            list of (neighbour_string, similarity_value) tuples.
            
        Raises:
            KeyError: if the word is not found in the semantic space.
            
        """
        
        start = time.time()
        assert_is_instance(similarity, Similarity)       
        vector = self.get_row(word)
        
        if space2 is None:
            id2row = self.id2row
            sims_to_matrix = similarity.get_sims_to_matrix(vector, 
                                                          self.cooccurrence_matrix)
        else:
            mat_type = type(space2.cooccurrence_matrix)
            if not isinstance(vector, mat_type):
                vector = mat_type(vector)
            
            sims_to_matrix = similarity.get_sims_to_matrix(vector, 
                                         space2.cooccurrence_matrix)
            id2row = space2.id2row 
        
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
        """
        Classmethod. Stacks two semantic spaces.
        
        The rows in the two spaces are concatenated.
            
        Args:
            space1, space2: spaces to be stacked, of type Space
            
        Returns:
            Stacked space, type Space.
            
        Raises:
            ValueError: if the spaces have different number of columns
                        or their columns are not identical
            
        """
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
        """
        Converts the matrix of the current space to DenseMatrix
        """
        self._cooccurrence_matrix = DenseMatrix(self.cooccurrence_matrix)
        
    def to_sparse(self):
        """
        Converts the matrix of the current space to SparseMatrix
        """
        self._cooccurrence_matrix = SparseMatrix(self.cooccurrence_matrix)
                
    def get_row(self, word):
        """
        Returns the row vector of a word.
        
        Args:
            word: string

        Returns: Matrix type (of shape (1, no_cols)), the row of the word argument.
        
        Raises:
            KeyError: if the word is not found in the space
        """
        return self.cooccurrence_matrix[self.row2id[word],:]
    
    def get_rows(self, words):
        """
        Returns the sub-matrix corresponding to a list of words.
        
        Args:
            words: list of strings

        Returns: Matrix type (of shape (len(words), no_cols)),
                 the sub-matrix containing the words given as an input.
        
        Raises:
            KeyError: if one of words is not found in the space
        """
        
        row_ids = []
        for word in words:
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
    """
    Co-occurrence matrix associated to the semantic space, of type Matrix.
    
    """    
    def get_row2id(self):
        return self._row2id
            
    row2id = property(get_row2id)
    """
    Dictionary, maps row strings to integer ids.
    """

    def get_id2row(self):
        return self._id2row
            
    id2row = property(get_id2row)
    """
    List of strings, the row elements.
    """
    def get_column2id(self):
        return self._column2id
            
    column2id = property(get_column2id)
    """
    Dictionary, maps column strings to integer ids.
    """

    def get_id2column(self):
        return self._id2column
            
    id2column = property(get_id2column)
    """
    List of strings, the column elements.
    """
       
    def get_element_shape(self):
        return self._element_shape
            
    element_shape = property(get_element_shape)
    """
    Shape of row elements, of type tuple. By default, in standard spaces, 
    element_shape=(no_cols,). 
    
    Used in composition models which build 
    word representations which are matrices or higher order tensors, instead
    of simple vectors. If the representation of a word is a matrix of shape
    (2,2) for example, then element_shape=(2,2). The actual space matrix
    stores each element as a linearized vector, just as in standard spaces.
    """ 
    
    def get_operations(self):
        return self._operations
            
    operations = property(get_operations)
    """
    List of operations which have been applied on the semantic space. List of
    Operation type objects.
    
    The operations, together with their associated side information, are stored
    because they may need to be projected on peripheral data.
    """
    
    def assert_1dim_element(self):
        """
        Asserts that the elements of the space are one dimensional.
        
        """
        if len(self.element_shape) > 1:
            raise IllegalOperationError("Operation not allowed on spaces with\
                                       element shape: %s" % self.element_shape)
   
    @classmethod
    def build(cls, **kwargs):
        """
        Reads in data files and extracts the data to construct a semantic space.
        
        If the data is read in dense format and no columns are provided, 
        the column indexing structures are set to empty.
         
        Args:
            data: file containing the counts
            format: format on the input data file: one of sm/dm
            rows: file containing the row elements. Optional, if not provided,
                extracted from the data file.
            cols: file containing the column elements
           
        Returns:
            A semantic space build from the input data files.
            
        Raises:
            ValueError: if one of data/format arguments is missing.
                        if cols is missing and format is "sm"
                        if the input columns provided are not consistent with
                        the shape of the matrix (for "dm" format)
            
        """
        start = time.time()
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
        
        if "rows" in kwargs and not kwargs["rows"] is None:
            [id2row], [row2id] = extract_indexing_structs(kwargs["rows"], [0])
            
        if "cols" in kwargs and not kwargs["cols"] is None:
            [id2column], [column2id] = extract_indexing_structs(kwargs["cols"], [0])
        elif format_ == "sm":
            raise ValueError("Need to specify column file when input format is sm!")
            
        if format_ == "sm":
            if id2row is None:
                [id2row], [row2id] = extract_indexing_structs(data_file, [0])
            mat = read_sparse_space_data(data_file, row2id, column2id)

        else:
            if id2row is None:
                [id2row],[row2id] = extract_indexing_structs(data_file, [0])
            mat = read_dense_space_data(data_file, row2id)
                
        if id2column and len(id2column) != mat.shape[1]:
            raise ValueError("Columns provided inconsistent with shape of input matrix!")

        if id2column is None:
            id2column, column2id = [], {}
                            
        log.print_matrix_info(logger, mat, 1, "Built semantic space:")
        log.print_time_info(logger, time.time(), start, 2)    
        return Space(mat, id2row, id2column, row2id, column2id)
    
    def export(self, file_prefix, **kwargs):
        """
        Exports the current space to disk.
        If the space has no column information, it cannot be exported in 
        sparse format (sm).
        
        Args:
            file_prefix: string, prefix of the files to be exported
            format: string, one of dm/sm
       
        Prints: 
            - matrix in file_prefix.<format> 
            - row elements in file_prefix.<row>
            - col elements in file_prefix.<col>
        
        Raises:
            ValueError: if the space has no column info and "sm" exporting
                is attempted
            NotImplementedError: the space matrix is dense and "sm" exporting
                is attempted
        
        """
        
        start = time.time()
        create_parent_directories(file_prefix)
        format_ = "dm"
        if "format" in kwargs:
            format_ = kwargs["format"]
            if not format_ in ["dm","sm"]:
                raise ValueError("Unrecognized format: %s" %format_)
            elif format_ == "dm":
                print_cooc_mat_dense_format(self.cooccurrence_matrix,
                                            self.id2row, file_prefix)
            else:
                print_cooc_mat_sparse_format(self.cooccurrence_matrix,
                                             self.id2row,
                                             self.id2column, file_prefix)
        self._export_row_column(file_prefix)
        
        log.print_matrix_info(logger, self.cooccurrence_matrix, 1, 
                              "Printed semantic space:")
        log.print_time_info(logger, time.time(), start, 2) 
        
    def _export_row_column(self, file_prefix):
        row_file = "%s.%s" %(file_prefix, "rows")
        column_file = "%s.%s" %(file_prefix, "cols")
        
        if self.column2id:
            print_list(self.id2column, column_file)
            
        print_list(self.id2row, row_file)



"""
def build(cls, **kwargs):
FANCY BUILD
        start = time.time()
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
        
        if "rows" in kwargs and not kwargs["rows"] is None:
            [id2row], [row2id] = extract_indexing_structs(kwargs["rows"], [0])
            
        if "cols" in kwargs and not kwargs["cols"] is None:
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
            
        log.print_matrix_info(logger, mat, 1, "Built semantic space:")
        log.print_time_info(logger, time.time(), start, 2)    
        return Space(mat, id2row, id2column, row2id, column2id)
FANCY BUILD


    
    Some transformations, such as weighings, only scale the values
    in the space matrix, while others, such as dimensionality
    reduction, or feature selection, alter the set of 
    contextual features.
"""
    