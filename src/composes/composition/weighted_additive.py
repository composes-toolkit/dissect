'''
Created on Oct 5, 2012

@author: georgianadinu
'''

from composition_model import CompositionModel
from composes.utils.num_utils import is_numeric
from numpy import double
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
    
    def __init__(self, alpha=None, beta=None):
        '''
        Constructor
        '''
        self._alpha = 0.5
        self._beta = 0.5
        if not alpha is None:
            if not is_numeric(alpha):
                raise ValueError("Parameter not numeric: %s " %(type(alpha)))
            else:
                self._alpha = alpha
                
        if not beta is None:
            if not is_numeric(beta):
                raise ValueError("Parameter not numeric: %s " %(type(beta)))
            else:
                self._beta = beta
                
        if not alpha is None and beta is None:
            self._beta = 1 - self._alpha    
            
        
    def _train(self, arg1_mat, arg2_mat, phrase_mat):    
        
        arg1_arg2_dot = arg1_mat.multiply(arg2_mat).sum()
        arg1_phrase_dot = arg1_mat.multiply(phrase_mat).sum()
        arg2_phrase_dot = arg2_mat.multiply(phrase_mat).sum()
        
        arg1_norm_sqr = pow(arg1_mat.norm(), 2)
        arg2_norm_sqr = pow(arg2_mat.norm(), 2)

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
    
    def get_beta(self):
        return self._beta
    beta = property(get_beta)
        
        