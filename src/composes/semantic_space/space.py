'''
Created on Sep 21, 2012

@author: georgianadinu
'''

from composes.utils.space_utils import list2dict
from composes.utils.space_utils import assert_dict_match_list
from composes.utils.space_utils import assert_shape_consistent
from composes.utils.space_utils import assert_is_instance 
from composes.matrix.matrix import Matrix
from composes.weighting.weighting import Weighting
from composes.operation import Operation

class Space(object):
    """
    This class implements semantic spaces.
    
    A semantic space contains a list of targets (words, phrases, etc.)
    described in terms of contextual features.
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

      
    def apply_operation(self, transformation):
        
        #TODO , FeatureSelection, DimReduction ..
                                            
        assert_is_instance(transformation, [Weighting])
        op = Operation(transformation)
        transformed_matrix =  op.apply(self.cooccurrence_matrix)
        transformed_operations = self.operations.copy()
        transformed_operations.append(op)
        #TODO
        #if isinstance(transformation, DimReduction) 
        #    return Space(_cooccurrence_matrix,... [],{})
        #elif isinstance(transformation, FeatureSelection)
        #    return Space(_cooccurrence_matrix,... [],{})
        #else:
        return Space(transformed_matrix, self.id2row.copy(),
                     self.id2column.copy(), self.row2id.copy(), 
                     self.column2id.copy(), transformed_operations)
        
        
    def set_cooccurrence_matrix(self, matrix_):
        assert_is_instance(matrix_, Matrix)
        self.assert_shape_consistent(matrix_, self.row2id, self.id2row,
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
    
    def get_operations(self):
        return self._operations
            
    operations = property(get_operations)
    