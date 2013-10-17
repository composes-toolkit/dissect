'''
Created on Oct 2, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

import numpy as np
from similarity import Similarity


class LinSimilarity(Similarity):
    """
    Computes the Lin similarity of two vectors.

    :math:`sim(\\vec{u},\\vec{v}) = \\frac{\\sum_{i \\in I}(u_i+v_i)}{\\sum_iu_i + \\sum_iv_i}`

    Where :math:`I=\\{i | u_i > 0 \\text{ and } v_i > 0\\}`, the set of components
    on which both vectors are strictly positive.

    """

    _name = "lin"

    def __init__(self):
        '''
        Constructor
        '''

    def _sim(self, v1, v2):

        common = v1.multiply(v2)
        common.to_ones()
        denom = v1.sum() + v2.sum()

        if denom == 0:
            return 0
        else:
            return common.multiply(v1 + v2).sum() / np.double(denom)


