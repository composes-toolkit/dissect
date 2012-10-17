'''
Created on Oct 5, 2012

@author: georgianadinu
'''
from composes.semantic_space.operation import FeatureSelectionOperation

class FeatureSelection(object):
    '''
    classdocs
    '''


    def __init__(self, reduced_dimension):
        
        if reduced_dimension <= 0:
            raise ValueError("Cannot reduce to non-positive dimensionality: %d"
                             % reduced_dimension)
        self._reduced_dimension = reduced_dimension

    def create_operation(self):
        return FeatureSelectionOperation(self)
    
    def get_reduced_dimension(self):
        return self._reduced_dimension
    
    reduced_dimension = property(get_reduced_dimension)        