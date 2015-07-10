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
        rowcount = matrix_.shape[0]
        matrix_type = type(matrix_)
        '''Returns a matrix of non-zero cells per row,
        a.k.a. the divisor of the IDF'''
        non_zero = (matrix_.get_mat() != 0).sum(1).flatten()
        idf = matrix_type(np.log(doccount / non_zero))
        idf.reshape((rowcount,1))
        matrix_ = matrix_type(matrix_.get_mat().toarray() *
                idf.get_mat().toarray())
        return matrix_

    def get_column_stats(self, matrix_):
        return matrix_.sum(0)
