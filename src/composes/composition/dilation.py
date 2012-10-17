'''
Created on Oct 15, 2012

@author: georgianadinu
'''
import numpy as np
from composition_model import CompositionModel
from composes.utils.num_utils import is_numeric
from composes.utils.py_matrix_utils import nonzero_invert


class Dilation(CompositionModel):
    '''
    classdocs
    '''

    _name = "dilation"
    
    def __init__(self, lambda_=None):
        '''
        Constructor
        '''
        self._lambda = 2
        if not lambda_ is None:
            if not is_numeric(lambda_):
                raise ValueError("Parameter not numeric: %s " %(type(lambda_)))
            else:
                self._lambda = lambda_

    def _train(self, arg1_mat, arg2_mat, phrase_mat):
        
        v2_minus_p = arg2_mat - phrase_mat
        
        v1_row_norms = arg1_mat.norm(1)
        v1_row_sqr_norms = np.multiply(v1_row_norms, v1_row_norms)
        inv_v1_row_sqr_norms = nonzero_invert(v1_row_sqr_norms)
        
        C = np.multiply(inv_v1_row_sqr_norms,
                        arg1_mat.multiply(arg2_mat).sum(1))
        nom = np.multiply(v1_row_sqr_norms, np.multiply(C,C)).sum()
        
        v1_dot_prod_v2_minus_p = arg1_mat.multiply(v2_minus_p).sum(1)
        denom = np.multiply(C, v1_dot_prod_v2_minus_p).sum()
 
        if nom != 0:
            self._lambda = 1 - denom/nom
        else:
            self._lambda = 2    

    def _compose(self, arg1_mat, arg2_mat):
        # TO DO: this is inefficient here, we do 2 for s instead of one
        # we do a for in get_rows in parent.compose() and a for here
        # comp = ((self._lambda -1) * v1.multiply(v2).sum()/pow(v1.norm(),2)) * v1 + v2
            
        v1_row_norms = arg1_mat.norm(1)    
        inv_v1_row_sqr_norms = nonzero_invert(np.multiply(v1_row_norms,
                                                          v1_row_norms))
        
        scale_factors = np.multiply(arg1_mat.multiply(arg2_mat).sum(1),
                                    inv_v1_row_sqr_norms)
        arg1_mat_scaled = arg1_mat.scale_rows(scale_factors)
            
        result = (self._lambda -1) * arg1_mat_scaled + arg2_mat
            
        return result
    
    def get_lambda(self):
        return self._lambda
    
    def _export(self, filename):
        with open(filename, "w") as output_stream:
            output_stream.write("lambda\t%f" % self._lambda)
