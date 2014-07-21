'''
Created on Oct 5, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

from .composition_model import CompositionModel
from composes.matrix.dense_matrix import DenseMatrix
from composes.utils.num_utils import is_numeric
# from composes.utils.mem_utils import get_mem_usage
from composes.utils.matrix_utils import resolve_type_conflict
import numpy as np

class WeightedAdditive(CompositionModel):
    """
    Implements weighted additive compositional model:

        :math:`\\vec{p} = \\alpha \\vec{u} + \\beta \\vec{v}`

    where :math:`\\vec{p}` is the vector of the composed phrase and
    :math:`\\vec{u}, \\vec{v}` are the vectors of the components

    When :math:`\\alpha=\\beta=0.5` the model performs simple vector addition.
    """

    _name = "weighted_additive"

    """
    double, in interval [0,1]
    maximum overhead allowed: MAX_MEM_OVERHEAD ratio of peripheral space memory
    """
    MAX_MEM_OVERHEAD = 0.2


    def __init__(self, alpha=None, beta=None):
        """
        Constructor.

        Args:
            alpha: alpha parameter, numeric type. Optional, can be set through
            training
            beta: beta parameter, numeric type. Optional, can be set through
            training.

        Raises:
            TypeError if alpha or beta are not numeric.
        """
        self._alpha = 0.5
        self._beta = 0.5
        if not alpha is None:
            if not is_numeric(alpha):
                raise TypeError("Parameter not numeric: %s " % (type(alpha)))
            else:
                self._alpha = alpha

        if not beta is None:
            if not is_numeric(beta):
                raise TypeError("Parameter not numeric: %s " % (type(beta)))
            else:
                self._beta = beta

        if not alpha is None and beta is None:
            self._beta = 1 - self._alpha

    def xxx(self, arg1_space, arg2_space, phrase_space, arg1_list, arg2_list, phrase_list):

        # we try to achieve at most MAX_MEM_OVERHEAD*phrase_space memory overhead
        # the /3.0 is needed
        # because the train data needs 3 * len(train_data) memory (arg1 vector, arg2 vector, phrase vector)
        chunk_size = int(phrase_space.cooccurrence_matrix.shape[0] * self.MAX_MEM_OVERHEAD / 3.0) + 1

        arg1_arg2_dot, arg1_phrase_dot, arg2_phrase_dot, arg1_norm_sqr, arg2_norm_sqr = (0, 0, 0, 0, 0)

        for i in range(int(len(arg1_list) // chunk_size)):
            beg, end = i*chunk_size, min((i+1)*chunk_size, len(arg1_list))

            arg1_mat = arg1_space.get_rows(arg1_list[beg:end])
            arg2_mat = arg2_space.get_rows(arg2_list[beg:end])
            phrase_mat = phrase_space.get_rows(phrase_list[beg:end])

            [arg1_mat, arg2_mat, phrase_mat] = resolve_type_conflict([arg1_mat,
                                                                      arg2_mat,
                                                                      phrase_mat],
                                                                      DenseMatrix)

            res = self._train1(arg1_mat, arg2_mat, phrase_mat)
            arg1_arg2_dot += res[0]
            arg1_phrase_dot += res[1]
            arg2_phrase_dot += res[2]
            arg1_norm_sqr += res[3]
            arg2_norm_sqr += res[4]


        self._train2(arg1_arg2_dot, arg1_phrase_dot, arg2_phrase_dot, arg1_norm_sqr, arg2_norm_sqr)


    def _train1(self, arg1_mat, arg2_mat, phrase_mat):

        # debug here
        # remove when done
        # print "Using %s MB " % (get_mem_usage())

        arg1_arg2_dot = arg1_mat.multiply(arg2_mat).sum()
        arg1_phrase_dot = arg1_mat.multiply(phrase_mat).sum()
        arg2_phrase_dot = arg2_mat.multiply(phrase_mat).sum()

        arg1_norm_sqr = pow(arg1_mat.norm(), 2)
        arg2_norm_sqr = pow(arg2_mat.norm(), 2)

        return arg1_arg2_dot, arg1_phrase_dot, arg2_phrase_dot, arg1_norm_sqr, arg2_norm_sqr

    def _train2(self, arg1_arg2_dot, arg1_phrase_dot, arg2_phrase_dot, arg1_norm_sqr, arg2_norm_sqr):

        a = np.linalg.pinv(np.mat([[arg1_norm_sqr,arg1_arg2_dot],
                                   [arg1_arg2_dot,arg2_norm_sqr]]))
        a = a * np.mat([[arg1_phrase_dot],[arg2_phrase_dot]])
        self._alpha = a[0, 0]
        self._beta = a[1, 0]


    def _compose(self, arg1_mat, arg2_mat):
        return self._alpha * arg1_mat + self._beta * arg2_mat

    def _export(self, filename):
        with open(filename, "w") as output_stream:
            output_stream.write("alpha\t%f\n" % self._alpha)
            output_stream.write("beta\t%f" % self._beta)

    def get_alpha(self):
        return self._alpha
    alpha = property(get_alpha)
    """
    Alpha parameter, default 0.5.
    """

    def get_beta(self):
        return self._beta
    beta = property(get_beta)
    """
    Beta parameter, default 0.5.
    """
