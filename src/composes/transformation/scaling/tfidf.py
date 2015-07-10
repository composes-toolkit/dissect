import numpy as np
from .scaling import Scaling
from .epmi_weighting import EpmiWeighting

class TfidfWeighting(Scaling):
    """
    Term Frequency -- Inverse Document Frequency


    :math:`tfidf(w,d,D) = P(w | d) * \log{frac{|D|}{|\{d\in{}D: w\in{} d\}|}}`
    """

    _name = "tfidf"
    _uses_column_stats = False

    def apply(self, matrix_):
        doccount = matrix_.shape[1]
        '''Returns a matrix of non-zero cells per row,
        a.k.a. the divisor of the IDF'''
        matrix_type = type(matrix_)
        non_zero = (matrix_.get_mat() != 0).sum(1).flatten()
        idf = matrix_type(np.log(doccount / non_zero))
        A=None
        for row, rowf in zip(matrix_, idf.transpose()):
            if not A:
                A = row*rowf[0,0]
            else:
                A = matrix_type.vstack(A, row*rowf[0,0])
        matrix_ = A
        return matrix_

    def get_column_stats(self, matrix_):
        return matrix_.sum(0)
