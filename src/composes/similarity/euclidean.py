'''
Created on Oct 2, 2012

@author: georgianadinu
'''

from similarity import Similarity
class EuclideanSimilarity(Similarity):
    '''
    classdocs
    '''
    _name = "euclidean"

    def __init__(self):
        '''
        Constructor
        '''
        
    def _sim(self, v1, v2):
        return 1 / (1 + (v1 - v2).norm())    