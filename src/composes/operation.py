'''
Created on Jun 6, 2012

@author: thenghia.pham
'''

#from weighting_scheme import Weighting
from composes.exception.illegal_state_error import IllegalStateError

# TODO: exception handling
class Operation(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        pass
    
"""
class ReductionOperation(Operation):
    
    def __init__(self, reduction_type, transformation_matrix):
        self.__trans_matrix = transformation_matrix
        self.__reduction_type = reduction_type
        
    def project_operation(self, matrix_):
        return matrix_ * self.__trans_matrix
    
    def apply()
    
    def __str__(self):
        return str(self.__reduction_type) + "_" + str(self.__trans_matrix.get_shape()[1])
"""
class WeightingOperation(Operation):
    """
    """
    def __init__(self, weighting):
        self.__weighting = weighting
        self.__column_stats = None

    def apply(self, matrix_):
      
        if not self.__column_stats is None:
            raise IllegalStateError("Illegal application of %s. Attempting\
                                     double application." % (self.__weighting))
        
        result_matrix = self.__weighting.apply(matrix_)
        
        if self.__weighting.uses_column_stats:
            self.__column_stats = self.__weighting.get_column_stats(matrix_)
        
        return result_matrix
    
    def project(self, matrix_):
        if self.__column_stats is None and self.__weighting.uses_column_stats:
            raise IllegalStateError("Illegal projection of %s. Attempting\
                                     projection before application." 
                                     % (self.__weighting))
        
        if self.__weighting.uses_column_stats:
            return self.__weighting.apply(matrix_, self.__column_stats)
        else:
            return self.__weighting.apply(matrix_)
    
    def __str__(self):
        return str(self.__weighting)
     
"""        
class FeatureSelectOperation       
        
    def __init__(self, weighting):
        self.__weighting = weighting

    def apply(self, matrix_):
        
        result_matrix = self__weighting.apply(matrix_)
        self.__column_marginal = matrix_.sum(0)
        
        return result_matrix
    
    def project(self, matrix_):
        return self.__weighting.apply(matrix_, self.__column_marginal)
    
    def __str__(self):
        return str(self.__weighting_scheme)  
"""        
 
        