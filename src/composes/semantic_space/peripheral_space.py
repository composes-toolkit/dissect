'''
Created on Sep 26, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

from space import Space
from numpy import array
from composes.utils.space_utils import list2dict
from composes.utils.space_utils import assert_dict_match_list
from composes.utils.space_utils import assert_shape_consistent
from composes.utils.space_utils import add_items_to_dict
from composes.semantic_space.operation import FeatureSelectionOperation
from composes.semantic_space.operation import DimensionalityReductionOperation
from composes.utils.gen_utils import assert_is_instance
from composes.matrix.matrix import Matrix

class PeripheralSpace(Space):
    '''
    classdocs
    '''


    def __init__(self, core_space, matrix_, id2row, row2id=None):
        """
        Constructor.

        Args:
            core_space: Space type, the core space that this is peripheral to.
            matrix_: Matrix type, the data matrix of the space
            id2row: list, the row elements
            row2id: dictionary, maps row strings to ids. Optional, built from
                id2row by default.

        Returns:
             A peripheral semantic space (type PeripheralSpace) on which the
             core space operations have been projected. Column indexing structures
             and operations are taken over from the core space.

        Raises:
            TypeError: if matrix_ or core_space are not of the correct type
            ValueError: if element shape is not consistent with
                         the size of matrix rows
                        if the matrix and the provided row and column
                         indexing structures are not of consistent shapes.
        """
        assert_is_instance(matrix_, Matrix)
        assert_is_instance(core_space, Space)
        assert_is_instance(id2row, list)
        # TODO: assert it is not a peripheral space here!

        if row2id is None:
            row2id = list2dict(id2row)
        else:
            assert_dict_match_list(row2id, id2row)

        column2id = core_space.column2id
        id2column = core_space.id2column

        self._operations = list(core_space.operations)
        self._row2id = row2id
        self._id2row = id2row
        self._column2id = column2id
        self._id2column = id2column

        self._cooccurrence_matrix = self._project_core_operations(matrix_)
        assert_shape_consistent(self.cooccurrence_matrix, self._id2row,
                                 self._id2column, self._row2id, self._column2id)

        self._element_shape = (self._cooccurrence_matrix.shape[1],)


    def _project_core_operations(self, matrix_):

        for operation in self._operations:
            if isinstance(operation, DimensionalityReductionOperation):
                self._id2column, self._column2id = [], {}

            if isinstance(operation, FeatureSelectionOperation):
                if operation.original_columns:
                    self._id2column = list(array(operation.original_columns)[operation.selected_columns])
                    self._column2id = list2dict(self._id2column)
                else:
                    self._id2column, self._column2id = [],{}

            matrix_ = operation.project(matrix_)
        return matrix_


    def add_rows(self, matrix_, id2row):
        """
        Adds rows to a peripheral space.

        Args:
            matrix_: Matrix type, the matrix of the elements to be added.
            id2row: list, string identifiers of the rows to be added.

        Modifies the current space by appending the new rows.
        All operations of the core space are projected to the new rows.

        Raises:
            ValueError: if attempting to add row strings which are already
                        in the space.
                        matrix of the new data is not consistent in shape
                        with the current data matrix.
        """

        try:
            self._row2id = add_items_to_dict(self.row2id, id2row)
        except ValueError:
            raise ValueError("Found duplicate keys when appending rows to\
                            peripheral space.")

        if matrix_.mat.shape[0] != len(id2row):
            raise ValueError("Matrix shape inconsistent with no. of rows:%s %s"
                              % (matrix_.mat.shape, len(id2row)))

        self._id2row = self.id2row + id2row
        matrix_ = self._project_core_operations(matrix_)

        self._cooccurrence_matrix = self._cooccurrence_matrix.vstack(matrix_)
        assert_shape_consistent(self.cooccurrence_matrix, self.id2row,
                                 self.id2column, self.row2id, self.column2id)

    @classmethod
    def build(cls, core_space, **kwargs):
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

        sp = Space.build(**kwargs)

        mat = sp._cooccurrence_matrix
        id2row = sp.id2row
        row2id = sp.row2id
        return PeripheralSpace(core_space, mat, id2row, row2id)




