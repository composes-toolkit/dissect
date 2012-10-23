'''
Created on Oct 4, 2012

@author: georgianadinu
'''
from numpy import double
from warnings import warn
from scaling import Scaling

class Normalization(Scaling):
    """
    Normalizes the a space according to a some criterion.
    
    Available criteria:
    
    sum: Default. The result matrix :math:`X` will satisfy: :math:`\\sum_{i,j} X_{ij}=1`  
    
    length: The result matrix :math:`X` will satisfy: :math:`\\sqrt{\\sum_{i,j} X_{ij}^2}=1`
    
    """
    _name = "row_normalization"
    _valid_criteria = ["sum", "length"]
    criterion = "sum"
    _uses_column_stats = True

    def __init__(self, **kwargs):
        '''
        Constructor
        '''
        if kwargs:
            if "criterion" in kwargs:
                criterion = kwargs["criterion"]
                if not criterion in self._valid_criteria:
                    raise ValueError("Unrecognized criterion: %s" % criterion)
                self.criterion = criterion
            else:
                raise ValueError("Unrecognized parameter: %s" % kwargs.keys()) 
        
        
    def apply(self, matrix_, total=None):
        
        if total is None:
            if self.criterion == "length":
                total = matrix_.norm()
            else:
                total = matrix_.sum()

        if total == 0:
            warn("Could not normalize: sum/length of matrix is 0.")
            return matrix_
                    
        matrix_ = (1 / double(total)) * matrix_
        return matrix_
    
    def get_column_stats(self, matrix_):
        
        if self.criterion == "length":
            return matrix_.norm()
        else:
            return matrix_.sum()
        