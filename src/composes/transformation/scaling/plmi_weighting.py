'''
Created on Sep 20, 2012

@author: georgianadinu
'''

from scaling import Scaling
from ppmi_weighting import PpmiWeighting

class PlmiWeighting(Scaling):
    '''
    classdocs
    '''
    _name = "plmi"
    _uses_column_stats = True

    def apply(self, matrix_, column_marginal=None):
        return matrix_.multiply(PpmiWeighting().apply(matrix_,
                                                                column_marginal))

    
    def get_column_stats(self, matrix_):
        return matrix_.sum(0)