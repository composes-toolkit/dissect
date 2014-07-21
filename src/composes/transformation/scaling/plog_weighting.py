
from .scaling import Scaling

class PlogWeighting(Scaling):
    """
     Positive Log Weighting

     :math:`plog(r,c)= log(r,c) \\text{ if } log(r,c) \\geq 0 \\text{ else } 0`
    """

    _name = "plog"

    def apply(self, matrix_):
        '''
        Performs positive log weighting.

        Args:
            matrix_ (Matrix): Input matrix
            column_marginal (array): column marginals of the core matrix if the matrix is a peripheral matrix

        Returns:
            Matrix: the matrix after applying plog

        '''
        matrix_ = matrix_.copy()
        matrix_.plog()
        return matrix_


