"""
Created on Oct 2, 2012

@author: Georgiana Dinu, Pham The Nghia
"""
import numpy as np

from composes.utils.py_matrix_utils import nonzero_invert

from composes.similarity.similarity import Similarity
from composes.similarity.dot_prod import DotProdSimilarity


class CosSimilarity(Similarity):
    """
    Computes the cosine similarity of two vectors.

    :math:`sim(\\vec{u},\\vec{v}) = \\frac{<\\vec{u},\\vec{v}>}{\\sqrt{||\\vec{u}||||\\vec{v}||}}`

    """

    def _sim(self, v1, v2):
        if v1.norm() == 0 or v2.norm() == 0:
            return 0.0
        s = DotProdSimilarity()._sim(v1, v2) / np.double(v1.norm() * v2.norm())
        return s

    def _sims_to_matrix(self, vector, matrix_):
        sims = DotProdSimilarity()._sims_to_matrix(vector, matrix_)

        vector_norm = vector.norm()
        row_norms = vector_norm * matrix_.norm(1)
        row_norms = nonzero_invert(row_norms)

        return sims.scale_rows(row_norms)




