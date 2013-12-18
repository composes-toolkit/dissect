
from scaling import Scaling
from ppmi_weighting import PpmiWeighting

class PlmiWeighting(Scaling):
    """
     Positive Local Mutual Information.

     :math:`plmi(r,c)=ppmi(r,c)count(r,c)`

    """

    _name = "plmi"
    _uses_column_stats = True

    def apply(self, matrix_, column_marginal=None):
        return matrix_.multiply(PpmiWeighting().apply(matrix_,
                                                                column_marginal))


    def get_column_stats(self, matrix_):
        return matrix_.sum(0)