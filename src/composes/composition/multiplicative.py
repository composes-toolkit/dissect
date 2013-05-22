'''
Created on Oct 5, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

from composition_model import CompositionModel
from composes.exception.illegal_state_error import IllegalOperationError

class Multiplicative(CompositionModel):
    """
    Implements the component-wise multiplication compositional model:
    
        :math:`\\vec{p} = \\vec{u} \\cdot \\vec{v}`
    
    where :math:`\\vec{p}` is the vector of the composed phrase and
    :math:`\\vec{u}, \\vec{v}` are the vectors of the components.
    
    :math:`\\vec{u} \\cdot \\vec{v} = (u_1v_1,...,u_nv_n)`  
    """
    
    _name = "multiplicative"

    def __init__(self):
        """
        Constructor
        """
        
    def train(self):
        """
        Current multiplicative model cannot be trained, it has no parameters.
        """    
        raise IllegalOperationError("Cannot train multiplicative model!")
    
    def _compose(self, arg1_mat, arg2_mat):
        return arg1_mat.multiply(arg2_mat)
    
    def export(self, filename):
        """
        Current multiplicative model cannot be exported, it has no parameters.
        """   
        raise IllegalOperationError("cannot export a Multiplicative model.")
            