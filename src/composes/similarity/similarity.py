"""
Created on Oct 2, 2012

@author: Georgiana Dinu, Pham The Nghia
"""
import numpy as np

from composes.utils.matrix_utils import (
    assert_is_array_or_matrix,
    to_compatible_matrix_types,
)


class Similarity(object):

    def get_sim(self, v1, v2):

        assert_is_array_or_matrix(v1)
        assert_is_array_or_matrix(v2)

        # TODO: figure out where these asserts belong!!
        v1, v2 = to_compatible_matrix_types(v1, v2)
        v1.assert_same_shape(v2)

        return self._sim(v1, v2)

    def get_sims_to_matrix(self, vector, matrix_):

        assert_is_array_or_matrix(vector)
        assert_is_array_or_matrix(matrix_)

        vector, matrix_ = to_compatible_matrix_types(vector, matrix_)

        if vector.shape[1] != matrix_.shape[1] or vector.shape[0] != 1:
            raise ValueError(
                'Inconsistent shapes {0} and {1}'.format(vector.shape, matrix_.shape)
            )

        return self._sims_to_matrix(vector, matrix_)

    def _sims_to_matrix(self, vector, matrix_):

        result = np.zeros(shape=(matrix_.shape[0], 1))
        for i in range(matrix_.shape[0]):
            result[i] = self._sim(vector, matrix_[i, :])
        return type(matrix_)(result)
