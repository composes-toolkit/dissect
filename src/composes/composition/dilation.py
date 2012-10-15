'''
Created on Oct 15, 2012

@author: georgianadinu
'''

from composition_model import CompositionModel
from composes.utils.num_utils import is_numeric
from composes.utils.matrix_utils import nonzero_invert
from composes.matrix.dense_matrix import DenseMatrix

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
        v1_row_norms_mat = DenseMatrix(v1_row_norms)
        v1_row_sqr_norms_mat = v1_row_norms_mat.multiply(v1_row_norms_mat)
        
        C = DenseMatrix((arg1_mat.multiply(arg2_mat).sum(1)/v1_row_norms)/v1_row_norms)
        
        nom = v1_row_sqr_norms_mat.multiply(C.multiply(C)).sum()
        
        v1_dot_prod_v2_minus_p = DenseMatrix(arg1_mat.multiply(v2_minus_p).sum(1))
        denom = C.multiply(v1_dot_prod_v2_minus_p).sum()
 
        if nom != 0:
            self._lambda = 1 - denom/nom

    def _compose(self, arg1_mat, arg2_mat):
        # TO DO: this is inefficient here, we do 2 for s instead of one
        # we do a for in get_rows in parent.compose() and a for here
        result_vecs = []
        for i in range(arg1_mat.shape[0]):
            v1 = arg1_mat[i,:]
            v2 = arg2_mat[i,:]
            comp = ((self._lambda -1) * v1.multiply(v2).sum()/pow(v1.norm(),2)) * v1 + v2
            
            result_vecs.append(comp)    
         
        result = type(comp).nary_vstack(result_vecs)        
        return result
    