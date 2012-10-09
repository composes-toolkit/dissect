'''
Created on Oct 5, 2012

@author: georgianadinu
'''

from composition_model import CompositionModel
from composes.utils.num_utils import is_numeric
from numpy import double
import numpy as np

class WeightedAdditive(CompositionModel):
    '''
    classdocs
    '''

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
         
        #alpha_denom = (arg2_norm_sqr * arg1_phrase_dot -
        #               arg1_arg2_dot * arg2_phrase_dot)
        #beta_denom = (arg1_norm_sqr * arg2_phrase_dot - 
        #              arg1_arg2_dot * arg1_phrase_dot)
        
        #nom = arg1_norm_sqr * arg2_norm_sqr - arg1_arg2_dot * arg1_arg2_dot
        
        #if the system is under-determined we use pinv to get a solution
        #if nom == 0:
        a = np.linalg.pinv(np.mat([[arg1_norm_sqr,arg1_arg2_dot],
                                       [arg1_arg2_dot,arg2_norm_sqr]]))
        a = a * np.mat([[arg1_phrase_dot],[arg2_phrase_dot]])
        self._alpha = a[0, 0]
        self._beta = a[1, 0]
            
        #return alpha_denom / double(nom), beta_denom / double(nom)
        
    def _compose(self, arg1_mat, arg2_mat):    
        return self._alpha * arg1_mat + self._beta * arg2_mat
        
    def get_alpha(self):
        return self._alpha
    alpha = property(get_alpha)
    
    def get_beta(self):
        return self._beta
    beta = property(get_beta)
        
        