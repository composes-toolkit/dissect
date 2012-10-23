'''
Created on Oct 2, 2012

@author: georgianadinu
'''

from similarity import Similarity
class EuclideanSimilarity(Similarity):
    """
    Computes the euclidean similarity of two vectors as the inverse of their
    euclidean distance.
        
    .. math:: sim(\\vec{u},\\vec{v}) = \\frac{1}{||\\vec{u}-\\vec{v}|| + 1}
    """
    
    _name = "euclidean"

    def __init__(self):
        '''
        Constructor
        '''
        
    def _sim(self, v1, v2):
        return 1 / (1 + (v1 - v2).norm())    