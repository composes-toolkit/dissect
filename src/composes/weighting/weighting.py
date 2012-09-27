'''
Created on Sep 20, 2012

@author: georgianadinu
'''

from composes.operation import WeightingOperation

class Weighting(object):
    '''
    classdocs
    '''
    _name = "we are NOT stupid"
    _uses_column_stats = False
        
    def get_name(self):
        return self._name
    
    def get_uses_column_stats(self):
        return self._uses_column_stats
    
    def create_operation(self):
        return WeightingOperation(self)
    
    def __str__(self):
        return self._name
    
    name = property(get_name)
    uses_column_stats = property(get_uses_column_stats)