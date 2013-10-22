"""
Created on Oct 2, 2012

@author: Georgiana Dinu, Pham The Nghia
"""
from composes.similarity.similarity import Similarity


class DotProdSimilarity(Similarity):
    """
    Computes the scalar product (dot product) of two vectors.

   :math:`sim(\\vec{u},\\vec{v}) = <\\vec{u},\\vec{v}> = \\sum_iu_iv_i`

    """
    def _sim(self, v1, v2):
        return v1.multiply(v2).sum()

    def _sims_to_matrix(self, vector, matrix_):
        return matrix_ * vector.transpose()
