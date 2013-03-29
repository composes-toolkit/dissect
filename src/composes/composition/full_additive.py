'''
Created on Oct 5, 2012

@author: georgianadinu
'''

from composition_model import CompositionModel
from composes.utils.space_utils import assert_is_instance
from composes.utils.matrix_utils import is_array_or_matrix
from composes.utils.matrix_utils import padd_matrix
from composes.utils.matrix_utils import to_compatible_matrix_types
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.utils.regression_learner import RegressionLearner
from composes.utils.matrix_utils import resolve_type_conflict
from composes.matrix.dense_matrix import DenseMatrix
from composes.exception.illegal_state_error import IllegalStateError

class FullAdditive(CompositionModel):
    """
    Implements the full additive compositional model: 
    
        :math:`\\vec{p} = A \\vec{u} + B \\vec{v}`
    
    where :math:`\\vec{p}` is the vector of the composed phrase,
    :math:`\\vec{u}, \\vec{v}`, the vectors of the components
    and :math:`A`, :math:`B` are two matrices.
          
    """
    _name = "full_additive"
    _mat_a_t = None
    _mat_b_t = None

    
    def __init__(self, **kwargs):
        #TODO here; very important, should be able to set the intercept
        #when mat a and mat b are given , to true or false. now by default is
        #is false
        """
        Constructor.
        
        Args:
            A= : matrix A, of matrix-like type (Matrix, ndarray, 
            numpy matrix, scipy matrix). Optional (parameters can be set
            through training.)
            
            B= : matrix B, matrix-like type. Optional.
        
            learner= : regression learner object, of type RegressionLearner.
            Optional, default LstsqRegressionLearner. 
        """
        if "A" in kwargs and "B" in kwargs:
            mat_a = kwargs["A"]
            mat_b = kwargs["B"]
            if not is_array_or_matrix(mat_a):
                raise TypeError("expected matrix type, received: %s" 
                                % type(mat_a))

            if not is_array_or_matrix(mat_b):
                raise TypeError("expected matrix type, received: %s" 
                                % type(mat_b))
                        
            mat_a, mat_b = to_compatible_matrix_types(mat_a, mat_b)                    
            self._mat_a_t = mat_a.transpose()
            self._mat_b_t = mat_b.transpose()
            self._has_intercept = False
            
        else:
            self._regression_learner = LstsqRegressionLearner()
            if "learner" in kwargs:
                self._regression_learner = kwargs["learner"] 
            self._has_intercept = self._regression_learner.has_intercept()
        
        
    def _train(self, arg1_mat, arg2_mat, phrase_mat):

        self._has_intercept = self._regression_learner.has_intercept()
        
        result = self._regression_learner.train(arg1_mat.hstack(arg2_mat), phrase_mat)

        self._mat_a_t = result[0:arg1_mat.shape[1], :]
        self._mat_b_t = result[arg1_mat.shape[1]:, :]

    
    def _compose(self, arg1_mat, arg2_mat):
        #NOTE when we get in this compose arg1 mat and arg2 mat have the same type
        [mat_a_t, mat_b_t, arg1_mat] = resolve_type_conflict([self._mat_a_t, 
                                                              self._mat_b_t, 
                                                              arg1_mat],
                                                             type(arg1_mat))   
        if self._has_intercept:
            return arg1_mat * mat_a_t + padd_matrix(arg2_mat, 1) * mat_b_t
        else:
            return arg1_mat * mat_a_t + arg2_mat * mat_b_t
        
    def set_regression_learner(self, regression_learner):
        assert_is_instance(regression_learner, RegressionLearner)
        self._regression_learner = regression_learner
        
    def get_regression_learner(self):
        return self._regression_learner
    
    regression_learner = property(get_regression_learner, set_regression_learner)    
    """
    Regression method to be used in training, of type RegressionLearner.
    Default is LstsqRegressionLearner.
    """
    
    def _build_id2column(self, arg1_space, arg2_space):
        return []

    def _export(self, filename):
        if self._mat_a_t is None or self._mat_b_t is None:
            raise IllegalStateError("cannot export an untrained FullAdditive model.")
            
        with open(filename, "w") as output_stream:
            output_stream.write("A\n")
            output_stream.write(str(DenseMatrix(self._mat_a_t).mat.T))
            output_stream.write("\nB\n")           
            
            if self._has_intercept:
                output_stream.write(str(DenseMatrix(self._mat_b_t[:-1,]).mat.T))
                output_stream.write("\nIntercept\n")
                output_stream.write(str(DenseMatrix(self._mat_b_t[-1,]).mat.T))
            else:
                output_stream.write(str(DenseMatrix(self._mat_b_t).mat.T))
           
    
    def get_mat_a_t(self):
        return self._mat_a_t
    mat_a_t = property(get_mat_a_t) 
    """
    Transpose of matrix A parameter, of type Matrix.
    """
            
    def get_mat_b_t(self):
        return self._mat_b_t
    mat_b_t = property(get_mat_b_t)
    """
    Transpose of matrix B parameter, of type Matrix.
    """
    