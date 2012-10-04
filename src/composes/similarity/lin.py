'''
Created on Oct 2, 2012

@author: georgianadinu
'''

import numpy as np
from similarity import Similarity

class LinSimilarity(Similarity):
    '''
    classdocs
    '''
    _name = "lin"

    def __init__(self):
        '''
        Constructor
        '''

    def _sim(self, v1, v2):

        common = v1.multiply(v2)
        common.to_ones()
        denom = v1.sum() + v2.sum()
        
        if denom == 0:
            return 0
        else:
            return common.multiply(v1 + v2).sum() / np.double(denom)
        
        