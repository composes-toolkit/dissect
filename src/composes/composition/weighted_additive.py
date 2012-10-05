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


    def __init__(self, alpha, beta):
        '''
        Constructor
        '''
        if not is_numeric(alpha) or not is_numeric(beta):
            raise ValueError("Parameters not numeric: %s %s" % 
                             (type(alpha), type(beta)))
        self._alpha = alpha
        self._beta = beta
        
    @classmethod    
    def _train(cls, arg1_mat, arg2_mat, phrase_mat):    
        
        arg1_arg2_dot = arg1_mat.multiply(arg2_mat).sum()
        arg1_phrase_dot = arg1_mat.multiply(phrase_mat).sum()
        arg2_phrase_dot = arg2_mat.multiply(phrase_mat).sum()
        
        arg1_norm_sqr = pow(arg1_mat.norm(), 2)
        arg2_norm_sqr = pow(arg2_mat.norm(), 2)
         
        alpha_denom = (arg2_norm_sqr * arg1_phrase_dot -
                       arg1_arg2_dot * arg2_phrase_dot)
        beta_denom = (arg1_norm_sqr * arg2_phrase_dot - 
                      arg1_arg2_dot * arg1_phrase_dot)
        
        nom = arg1_norm_sqr * arg2_norm_sqr - arg1_arg2_dot * arg1_arg2_dot
        
        #if the system is under-determined we use pinv to get a solution
        if nom == 0:
            a = np.linalg.pinv(np.mat([[arg1_norm_sqr,arg1_arg2_dot],
                                       [arg1_arg2_dot,arg2_norm_sqr]]))
            a = a * np.mat([[arg1_phrase_dot],[arg2_phrase_dot]])
            return a[0, 0], a[1, 0]
            
        return alpha_denom / double(nom), beta_denom / double(nom)
        
    
    def _compose(self, arg1_mat, arg2_mat):    
        return self._alpha * arg1_mat + self._beta * arg2_mat
        
    def get_alpha(self):
        return self._alpha
    alpha = property(get_alpha)
    
    def get_beta(self):
        return self._beta
    beta = property(get_beta)
        
        