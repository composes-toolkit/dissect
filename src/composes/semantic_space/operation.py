'''
Created on Jun 6, 2012

@author: thenghia.pham
'''

from composes.matrix.dense_matrix import DenseMatrix
from composes.exception.illegal_state_error import IllegalStateError
from composes.utils.matrix_utils import resolve_type_conflict
from warnings import warn

class Operation(object):
    """
    This class implements both the application, and the projection of a 
    transformation on a semantic space.
    
    An operation object can be used to apply or to project a specific
    transformation on a semantic space. 
    After a transformation is applied, for example on a core space, the operation 
    class stores the information required to further project this same 
    operation onto a space peripheral to the core space.  
    """

    def __init__(self):
        """
        Constructor
        """
        pass
    
    def _raise_projection_error(self, transformation):
        raise IllegalStateError("Illegal projection of %s. Attempting\
                                 projection before application." 
                                 % (transformation))        
    
    def _raise_double_application_error(self, transformation):
        raise IllegalStateError("Illegal application of %s. Attempting\
                                     double application." % (transformation))
        
class ScalingOperation(Operation):
    """
    This class implements the application and the projection of scaling
    transformations.
    """
    
    def __init__(self, scaling):
        self.__scaling = scaling
        self.__column_stats = None

    def apply(self, matrix_):
        """
        Applies a scaling operation.
        
        Args:
            matrix_: matrix on which the scaling is applied, of type Matrix
        
        Returns:
            the scaled matrix
        
        The column statistics computed by the scaling transformation, if any,
        is stored in the current operation object. For example, PPMI scaling
        needs column sums in order to be projected on peripheral spaces, 
        while PLOG scaling does not require this.
            
        """
        
        if not self.__column_stats is None:
            self._raise_double_application_error(self.__scaling)
        
        result_matrix = self.__scaling.apply(matrix_)
        
        if self.__scaling.uses_column_stats:
            self.__column_stats = self.__scaling.get_column_stats(matrix_)
        
        return result_matrix
    
    def project(self, matrix_):
        """
        Projects a scaling operation.
        
        Args:
            matrix_: matrix on which the scaling is projected, of type Matrix
        
        Returns:
            the scaled matrix
        
        If the current operation object has column_stats, this structure is
        used in the projection.
        """
        
        if self.__column_stats is None and self.__scaling.uses_column_stats:
            self._raise_projection_error(self.__scaling)
        
        if self.__scaling.uses_column_stats:
            return self.__scaling.apply(matrix_, self.__column_stats)
        else:
            return self.__scaling.apply(matrix_)
    
    def __str__(self):
        return str(self.__scaling)
     
     
class DimensionalityReductionOperation(Operation):
    """
    This class implements the application and the projection of dimensionality
    reduction transformations.
    """
    
    def __init__(self, dim_reduction):
        self.__dim_reduction = dim_reduction
        self.__transmat = None
        
    def apply(self, matrix_):
        """
        Applies a dim. reduction operation.
        
        Args:
            matrix_: matrix on which the reduction is applied, of type Matrix
        
        Returns:
            the reduced matrix
        
        The transformation matrix obtained in the reduction (specific to each
        reduction method) is stored in the operation object. This transformation
        matrix is further used for projecting the dim. reduction method on 
        a space peripheral to the space on which it has been originally applied.
        """
                   
        if not self.__transmat is None:
            self._raise_double_application_error(self.__dim_reduction)
        
        res_mat, self.__transmat = self.__dim_reduction.apply(matrix_)
        
        return DenseMatrix(res_mat)         
     
    def project(self, matrix_):
        """
        Projects a dim. reduction operation.
        
        Args:
            matrix_: matrix on which the reduction is projected, of type Matrix
        
        Returns:
            the reduced matrix
        
        Uses the transformation matrix stored in the operation object to project
        the dimensionality reduction method on a new space, peripheral to the
        original one.
        """
        
        if self.__transmat is None:
            self._raise_projection_error(self.__dim_reduction)
        
        if self.__dim_reduction.name == "nmf":
            matrix_.assert_positive()
        
        if not isinstance(matrix_, type(self.__transmat)):
            warn("WARNING: peripheral matrix type (dense/sparse) should be the same as the core space matrix type!!")
        
        [matrix_, transmat] = resolve_type_conflict([matrix_, self.__transmat], 
                                                        type(matrix_))
        
        result_mat = matrix_ * transmat
         
        if self.__dim_reduction.name == "nmf":
            result_mat.to_non_negative()
                
        return DenseMatrix(result_mat)

    def __str__(self):
        return str(self.__dim_reduction)        
             
      
class FeatureSelectionOperation(Operation):       
    """
    This class implements the application and the projection of feature 
    selection transformations.
    """
            
    def __init__(self, feat_selection):
        self.__feat_selection = feat_selection
        self.__selected_columns = None
        self.__original_columns = None

    def apply(self, matrix_):
        """
        Applies a dim. feature selection operation.
        
        Args:
            matrix_: matrix on which the reduction is applied, of type Matrix
        
        Returns:
            the reduced matrix
        
        The columns selected are stored in the operation object. These are 
        further used for projecting the feature selection method on 
        a space peripheral to the original space on which it has been applied.
        """
                
        if not self.__selected_columns is None:
            self._raise_double_application_error(self.__feat_selection)
        
        res_mat, self.__selected_columns = self.__feat_selection.apply(matrix_)
        return res_mat
    
    def project(self, matrix_):
        """
        Projects a feature selection operation.
        
        Args:
            matrix_: matrix on which the selection is applied, of type Matrix
        
        Returns:
            the reduced matrix
        
        Uses the information on selected columns stored in the operation 
        object to project the feature selection method on a new space, 
        peripheral to the original one.
        """        
        
        if self.__selected_columns is None:        
            self._raise_projection_error(self.__dim_reduction)
                    
        res_mat = matrix_[:, self.__selected_columns]
        return res_mat
    
    
    def __str__(self):
        return str(self.__feat_selection)  
      
 
    def get_selected_columns(self):
        return self.__selected_columns

    def get_original_columns(self):
        return self.__original_columns
    
    def set_original_columns(self, original_columns):
        self.__original_columns = original_columns
        
    selected_columns = property(get_selected_columns)
    """
    List of integers, indices of the columns selected.
    """    
    original_columns = property(get_original_columns, set_original_columns)   
    """
    List of strings, the id2column of the space before applying the 
    feature selection.
    """    
