
from scaling import Scaling
from epmi_weighting import EpmiWeighting

class PpmiWeighting(Scaling):
    """
    Positive Point-wise Mutual Information.
    
     
    :math:`pmi(r,c) = log\\frac{P(r,c)}{P(r)P(c)}`
     
    :math:`ppmi(r,c)= pmi(r,c) \\text{ if } pmi(r,c)\\geq 0 \\text{ else } 0`         
    """
    
    _name = "ppmi"
    _uses_column_stats = True

    def apply(self, matrix_, column_marginal=None):
        
        matrix_ = EpmiWeighting().apply(matrix_, column_marginal)
        matrix_.plog()
        return matrix_
    
    def get_column_stats(self, matrix_):
        return matrix_.sum(0)
    
    """
    :math:`ppmi(r,c)=\\begin{cases}pmi(rc) & \\text{if }pmi(r,c)\\geq0 
                      0 & \\text{otherwise}\\end{cases}`         
    """