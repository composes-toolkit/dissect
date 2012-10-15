'''
Created on Oct 15, 2012

@author: georgianadinu
'''

from composition_model import CompositionModel
from composes.utils.num_utils import is_numeric
from composes.utils.matrix_utils import nonzero_invert

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
        
        dot_prod_arg1_ph = arg1_mat.multiply(phrase_mat).sum()
        norm_arg1 = pow(arg1_mat.norm(),2)

        #row_normalize matrix u
        row_norms = arg1_mat.norm(axis=1)
        inv_row_norm = nonzero_invert(row_norms)
        row_normd_arg1 = arg1_mat.scale_rows(inv_row_norm)
        
        dot_prod_arg1_arg2 = row_normd_arg1.multiply(arg2_mat).sum()
        
        if norm_arg1 * dot_prod_arg1_arg2 == -1:
            self._lambda = 2
        else:    
            self._lambda = dot_prod_arg1_ph/(norm_arg1 * dot_prod_arg1_arg2) + 1
        

    def _compose(self, arg1_mat, arg2_mat):
        # TO DO: this is inefficient here, we do 2 for s instead of one
        # we do a for in get_rows in parent.compose() and a for here
        result_vecs = []
        for i in range(arg1_mat.shape[0]):
            v1 = arg1_mat[i]
            v2 = arg2_mat[i]
            comp = ((self._lambda -1) * v1.multiply(v2).sum()/pow(v1.norm(),2)) * v1 + v2
            
            result_vecs.append(comp)    
         
        result = type(comp).vstack(result_vecs)        
        return result
    