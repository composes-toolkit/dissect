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
     
     
class DimensionalityReductionOperation(Operation):
     
    def __init__(self, dim_reduction):
        self.__dim_reduction = dim_reduction
        self.__transmat = None
        
    def apply(self, matrix_):
      
        if not self.__transmat is None:
            raise IllegalStateError("Illegal application of %s. Attempting\
                                     double application." 
                                     % (self.__dim_reduction))
        
        res_mat, self.__transmat = self.__dim_reduction.apply(matrix_)
        
        return res_mat         
     
    def project(self, matrix_):
        if self.__transmat is None:
            raise IllegalStateError("Illegal projection of %s. Attempting\
                                     projection before application." 
                                     % (self.__dim_reduction))
        
        
        if self.__dim_reduction.name == "nmf":
            matrix_.assert_positive()
        
        result_mat = matrix_ * self.__transmat
         
        if self.__dim_reduction.name == "nmf":
            result_mat.to_non_negative()
                
        return result_mat
        
             
      
class FeatureSelectionOperation(Operation):       
        
    def __init__(self, feat_selection):
        self.__feat_selection = feat_selection
        self.__selected_columns = None

    def apply(self, matrix_):
        
        if not self.__selected_columns is None:
            raise IllegalStateError("Illegal application of %s. Attempting\
                                     double application." 
                                     % (self.__feat_selection))
        
        res_mat, self.__selected_columns = self.__feat_selection.apply(matrix_)
        return res_mat
    
    def project(self, matrix_):
        
        res_mat = matrix_[:, self.__selected_columns]
        return res_mat
    
    
    def __str__(self):
        return str(self.__weighting_scheme)  
      
 
    def get_selected_columns(self):
        return self.__selected_columns
    
    selected_columns = property(get_selected_columns)    
        