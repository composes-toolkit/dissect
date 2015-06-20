
from .scaling import Scaling
from .epmi_weighting import EpmiWeighting

class TfidfWeighting(Scaling):
    """
    Term Frequency -- Inverse Document Frequency


    :math:`tfidf(w,d,D) = P(w | d) * \log{frac{|D|}{|\{d\in{}D: w\in{} d\}|}}`
    """

    _name = "tfidf"
    _uses_column_stats = True

    def apply(self, matrix_):
        colsums = np.sum(matrix_, axis=0)
        doccount = mat.shape[1]
        matrix_ = np.array([TfidfWeighting._tfidf_row_func(row, colsums, doccount) for row in matrix_])
        return matrix_

    @staticmethod
    def _tfidf_row_func(row, colsums, doccount):
        df = float(len([x for x in row if x > 0]))
        idf = 0.0
        if df > 0.0 and df != doccount:
            idf = np.log(doccount / df)
        tfs = row/colsums
        return tfs * idf

    def get_column_stats(self, matrix_):
        return matrix_.sum(0)

    """
    :math:`ppmi(r,c)=\\begin{cases}pmi(rc) & \\text{if }pmi(r,c)\\geq0
                      0 & \\text{otherwise}\\end{cases}`
    """
