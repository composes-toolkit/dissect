'''
Created on Oct 5, 2012

@author: georgianadinu
'''

from composition_model import CompositionModel
from numpy import linspace
from composes.matrix.linalg import Linalg  
from composes.utils.space_utils import assert_is_instance

class FullAdditive(CompositionModel):
    '''
    classdocs
    '''
    _name = "full_additive"
    
    def __init__(self, mat_a=None, mat_b=None, ):
        '''
        Constructor
        '''
        if not mat_a is None and not mat_b is None:
            self._mat_a_t = mat_a.transpose()
            self._mat_b_t = mat_b.transpose()
            self._has_intercept = False
        else:
                
            self._regression_learner = learner
            self._has_intercept = learner.has_intercept
        
           
        
    def _train(self, arg1_mat, arg2_mat, phrase_mat):

        self._has_intercept = self._regression_learner.has_intercept
        
        result = self._regression_learner.train(arg1_mat.hstack(arg2_mat), phrase_mat)

        self._mat_a_t = result[0:arg1_mat.shape[1], :]
        self._mat_b_t = result[arg1_mat.shape[1]:, :]

    
    def _compose(self, arg1_mat, arg2_mat):
        # TODO tomorrow 
        if self._has_intercept:
            return arg1_mat * self._mat_a_t + padd_ones(arg2_mat, 1) * self._mat_b_t
        else:
            return arg1_mat * self._mat_a_t + arg2_mat * self._mat_b_t
        
        
        
    
    def set_regression_learner(self, regression_learner):
        assert_is_instance(regression_learner, RegressionLearner)
        self._regression_learner = regression_learner
        
        
    def get_regression_learner(self):
        return self._regression_learner
    
    regression_learner = property(get_regression_learner, set_regression_learner)    
    """
    
            
        #if cls._regression_method == Linalg.ridge_regression:
        
        #    lambdas = linspace(0.1,0.1,30)
        #    result = learningutils.crossvalidation_train(arg1_mat.hstack(arg2_mat), 
        #                                    phrase_mat, lambdas)
                                                    
        #else:     
        
    FullAdditive._regression_method = learning_utils.regression_with_cross_validation(ridge_regression, lambdas)._regression
    
    
    
    result = cls._regression_method(arg1_mat.hstack(arg2_mat), phrase_mat)
    
    #1
    
    
        
        if has_parameters(cls._regression_method):
        
            lambdas = linspace(0.1,0.1,30)
            if lambda_ is None:
                lambda_ = learningutils.get_crossvalidation_param(cls._regression_method, arg1_mat.hstack(arg2_mat), 
                                            phrase_mat, lambdas)
            
            result = cls._regression_method(arg1_mat.hstack(arg2_mat), 
                                             phrase_mat, lambda_)
                                                    
        else:            
            result = cls._regression_method(arg1_mat.hstack(arg2_mat), phrase_mat)
        
        
        
        #2
        
        if has_parameters(cls._regression_method):
        
            lambdas = linspace(0.1,0.1,30)
            result = learningutils.crossvalidation_train(arg1_mat.hstack(arg2_mat), 
                                            phrase_mat, lambdas)

        else:            
            result = learningutils.train(arg1_mat.hstack(arg2_mat), phrase_mat)
        
        
       """ 
        
    
    def _compose(self, arg1_mat, arg2_mat):    
        return self._mat_a * arg1_mat + self._mat_b * arg2_mat
    
    def _build_id2column(self, arg1_space, arg2_space):
        return []

    @classmethod
    def get_regression_method(cls):
        return cls._regression_method
    
    @classmethod
    def set_regression_method(cls, regression_function):
        cls._regression_method = regression_function
        
            
    regression_method = property(get_regression_method, 
                                      set_regression_method) 
    
