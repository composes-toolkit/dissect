'''
Created on Oct 5, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
from warnings import warn
from .feature_selection import FeatureSelection

class TopFeatureSelection(FeatureSelection):
    """
    Sorts the columns of a space according to some criterion and returns a space
    containing only the top :math:`k` ones.

    Available criteria:

    sum: Default. Ranks columns according to the sum on their elements.

    length: Ranks columns according to their vector length.

    """

    _name = "top_feature_selection"
    _valid_criteria = {"sum", "length"}

    def __init__(self, reduced_dimension, criterion='sum'):
        '''
        Constructor
        '''
        super(TopFeatureSelection, self).__init__(reduced_dimension)

        if criterion:
            if criterion not in self._valid_criteria:
                raise ValueError("Unrecognized criterion: %s" % criterion)
            self.criterion = criterion

    def apply(self, matrix_):

        if self.criterion == "sum":
            norm_function = matrix_.sum
        else:
            norm_function = matrix_.norm

        if self._reduced_dimension >= matrix_.shape[1]:
            warn("Reduced dimension larger than number of columns!")

        no_columns = min(self._reduced_dimension, matrix_.shape[1])
        sorted_perm = matrix_.sorted_permutation(norm_function, 0)

        sorted_perm = sorted_perm[0:no_columns]
        matrix_ = matrix_[:, sorted_perm]

        return matrix_, sorted_perm


