'''
Created on Oct 15, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import numpy as np
from composition_model import CompositionModel
from composes.utils.num_utils import is_numeric
from composes.utils.py_matrix_utils import nonzero_invert


class Dilation(CompositionModel):
    """
    Implements the dilation compositional model:

        :math:`\\vec{p} = (\\vec{u} \\cdot \\vec{u}) \\vec{v} + (\\lambda - 1) (\\vec{u} \\cdot \\vec{v}) \\vec{u}`
    
    where :math:`\\vec{p}` is the vector of the composed phrase, :math:`\\vec{u}, \\vec{v}` the vectors of the components
    and :math:`\\lambda` is a scalar.

    """


    _name = "dilation"
    
    _lambda = 2

            
    def __init__(self, lambda_=None):
        """
        Constructor.
        
        Args:
            lambda_ : numeric, value of the lambda parameter. Optional.
        """

        if not lambda_ is None:
            if not is_numeric(lambda_):
                raise ValueError("Parameter not numeric: %s " %(type(lambda_)))
            else:
                self._lambda = lambda_

    def _train(self, arg1_mat, arg2_mat, phrase_mat):

        v1_row_norms = arg1_mat.norm(1)
        v1_row_sqr_norms = np.multiply(v1_row_norms, v1_row_norms)

        v2_minus_p = arg2_mat.scale_rows(v1_row_sqr_norms) - phrase_mat
        v1_dot_prod_v2_minus_p = arg1_mat.multiply(v2_minus_p).sum(1)
        
        v1_v2 = arg1_mat.multiply(arg2_mat).sum(1)
        v1_v2_sqr = np.multiply(v1_v2, v1_v2)
        
        nom = np.multiply(v1_v2_sqr, v1_row_sqr_norms).sum()
        denom = np.multiply(v1_v2, v1_dot_prod_v2_minus_p).sum()
 
        if nom != 0:
            self._lambda = 1 - denom/nom
        else:
            self._lambda = 2
                   

    def _compose(self, arg1_mat, arg2_mat):
        # TO DO: this is inefficient here, we do 2 for s instead of one
        # we do a for in get_rows in parent.compose() and a for here
        # comp = ((self._lambda -1) * v1.multiply(v2).sum()/pow(v1.norm(),2)) * v1 + v2
            
        v1_row_norms = arg1_mat.norm(1)    
        scale_factors1 = arg1_mat.multiply(arg2_mat).sum(1)
        scale_factors2 = np.multiply(v1_row_norms, v1_row_norms)
                         
        arg1_mat_scaled = arg1_mat.scale_rows(scale_factors1)
        arg2_mat_scaled = arg2_mat.scale_rows(scale_factors2)
        
        #print "FACTORS u:", ((self._lambda -1)*scale_factors1).sum()/float(len(scale_factors1))
        #print "FACTORS v:", (scale_factors2).sum()/float(len(scale_factors2))
            
        result = (self._lambda - 1) * arg1_mat_scaled + arg2_mat_scaled
            
        return result
    
    def get_lambda(self):
        return self._lambda
    """
    Lambda parameter. Default, set to lambda=2.
    """
        

    def _export(self, filename):
        with open(filename, "w") as output_stream:
            output_stream.write("lambda\t%f" % self._lambda)
