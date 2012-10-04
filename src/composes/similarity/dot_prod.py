'''
Created on Oct 2, 2012

@author: georgianadinu
'''

from similarity import Similarity
class DotProdSimilarity(Similarity):
    '''
    classdocs
    '''
    _name = "dot_prod"

    def __init__(self):
        '''
        Constructor
        '''
        
    def _sim(self, v1, v2):
        return v1.multiply(v2).sum()
    
    def _sims_to_matrix(self, vector, matrix_):
        return matrix_ * vector.transpose()    