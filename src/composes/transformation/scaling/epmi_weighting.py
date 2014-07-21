
from .scaling import Scaling
from composes.utils.py_matrix_utils import nonzero_invert

class EpmiWeighting(Scaling):
    """
     Exponential Point-wise Mutual Information.

     :math:`epmi(r,c) = \\frac{P(r,c)}{P(r)P(c)}`

    """

    _name = 'epmi'
    _uses_column_stats = True

    def apply(self, matrix_, column_marginal=None):
        """
        Performs epmi weighting.

        Args:
            matrix_ (Matrix): Input matrix

            column_marginal (np.ndarray): column marginals of the
                core matrix if the matrix is a peripheral matrix

        Returns:
            Matrix: the matrix after applying epmi.

        """

        matrix_.assert_positive()
        row_sum = matrix_.sum(axis = 1)

        if not column_marginal is None:
            col_sum = column_marginal
        else:
            col_sum = matrix_.sum(axis = 0)

        total = col_sum.sum()

        row_sum = nonzero_invert(row_sum)
        col_sum = nonzero_invert(col_sum)
        col_sum = col_sum * total

        matrix_ = matrix_.scale_rows(row_sum)
        matrix_ = matrix_.scale_columns(col_sum)

        return matrix_

    def get_column_stats(self, matrix_):
        return matrix_.sum(0)

