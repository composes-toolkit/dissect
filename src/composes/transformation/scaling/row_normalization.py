'''
Created on Oct 4, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

from scaling import Scaling
from composes.utils.py_matrix_utils import nonzero_invert

class RowNormalization(Scaling):
    """
    Normalizes the rows of a space according to a some criterion.

    Available criteria:

    length: Default. Each row :math:`X_i` of the result matrix will satisfy: :math:`\\sqrt{\\sum_j X_{ij}^2}=1`


    sum: Each row :math:`X_i` of the result matrix will satisfy: :math:`\\sum_j X_{ij}=1`

    """
    _name = "row_normalization"
    _valid_criteria = ["sum", "length"]

    def __init__(self, criterion='length'):
        '''
        Constructor
        '''
        if criterion:
            if criterion not in self._valid_criteria:
                raise ValueError("Unrecognized criterion: %s" % criterion)
            self.criterion = criterion


    def apply(self, matrix_):

        if self.criterion == "length":
            row_norms = matrix_.norm(axis=1)
        else:
            row_norms = matrix_.sum(axis=1)

        inv_row_norm = nonzero_invert(row_norms)
        matrix_ = matrix_.scale_rows(inv_row_norm)
        return matrix_


