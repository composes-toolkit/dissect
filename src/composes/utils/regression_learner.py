import numpy as np
from composes.matrix.linalg import Linalg
from crossvalidation_utils import get_split_indices
from crossvalidation_utils import get_submatrix_list
from matrix_utils import padd_matrix 

class RegressionLearner(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''

    def has_intercept(self):
        return self._intercept    
        
class LstsqRegressionLearner(RegressionLearner):
        
    """
    This class performs Least Squares Regression.
        
    It finds the matrix X which solves:
    
    :math:`X = argmin(||AX - B||)`
    
    It can be used with intercept or without (by default intercept=True).
    
    """    
    def __init__(self, **kwargs):
        
        self._intercept = True
        if "intercept" in kwargs:
            self._intercept = kwargs["intercept"]
                
    def train(self, matrix_a, matrix_b):
        return Linalg.lstsq_regression(matrix_a, matrix_b, self._intercept)
           
        
class RidgeRegressionLearner(RegressionLearner):
    
    """
    This class performs Ridge Regression.
        
    It finds the matrix X which solves:
    
    :math:`X = argmin(||AX - B|| + \\lambda||X||)`
    
    It can be used with intercept or without (by default intercept=True).
    Crossvalidation can be used with default :math:`\\lambda` range of 
    :math:`linspace(0, 0.5, 10)`. By default 10-fold crossvalidation is performed.
    If crossvalidation is set False it requires the input of a :math:`\\lambda` value.
    
    """ 
        
    def __init__(self, **kwargs):

        self._intercept = True
        if "intercept" in kwargs:
            self._intercept = kwargs["intercept"]
        
        self._crossvalidation = True
        
        self._folds = 10
        if "folds" in kwargs:
            if not kwargs["folds"] is None: 
                self._folds = kwargs["folds"] 
            
        self._param_range = np.linspace(0.0, 0.5, 10)
        if "param_range" in kwargs:
            if not kwargs["param_range"] is None:
                self._param_range = kwargs["param_range"]
        
        self._param = None
        if "param" in kwargs:
            if not kwargs["param"] is None:
                self._crossvalidation = False
                self._param = kwargs["param"]

        if "crossvalidation" in kwargs:
            self._crossvalidation = kwargs["crossvalidation"]
                    
        if not self._crossvalidation and self._param is None:
            raise ValueError("Cannot run (no-crossvalidation) RidgeRegression with no lambda value!")
        
          
    def train(self, matrix_a, matrix_b):
        
        if not self._crossvalidation:
            return Linalg.ridge_regression(matrix_a, matrix_b, self._param, 
                                     self._intercept)
         
        else:
            submatrix_indices = get_split_indices(matrix_a.shape[0], 
                                                  self._folds) 
            no_submatrices = len(submatrix_indices)
            if no_submatrices <= 1:
                raise ValueError("Cannot perform crossvalidation with one sample")
            
            submat_list_a = get_submatrix_list(matrix_a, submatrix_indices)
            submat_list_b = get_submatrix_list(matrix_b, submatrix_indices)
            
            min_err_param = 0 
            min_err = np.Inf
            
            for param in self._param_range:
                sum_err = 0
                for i in range(no_submatrices):
                    test_submat_a = submat_list_a[i]
                    test_submat_b = submat_list_b[i]
                    
                    train_submat_a = matrix_a.nary_vstack([submat_list_a[j] for j 
                                             in range(no_submatrices) if j != i])
                     
                    train_submat_b = matrix_b.nary_vstack([submat_list_b[j] for j 
                                             in range(no_submatrices) if j != i])
                    
                    mat_x = Linalg.ridge_regression(train_submat_a, train_submat_b, 
                                            param, self._intercept)
                    
                    if self._intercept:
                        test_submat_a = padd_matrix(test_submat_a, 1)
                           
                    err = (test_submat_a * mat_x - test_submat_b).norm()   
                    sum_err += err * err
                
                mean_sqr_err = sum_err / np.double(no_submatrices)
                if mean_sqr_err < min_err:
                    min_err = mean_sqr_err
                    min_err_param = param
                
            return Linalg.ridge_regression(matrix_a, matrix_b, min_err_param, 
                                     self._intercept)  
                