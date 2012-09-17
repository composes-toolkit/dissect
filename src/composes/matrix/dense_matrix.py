'''
Created on Sep 17, 2012

@author: georgianadinu
'''

import numpy as np
from warnings import warn
from composes.matrix.matrix import Matrix

class DenseMatrix(object):
    '''
    classdocs
    '''

    def __init__(self, data, *args, **kwargs):
        '''
        Constructor, create a Dense from a numpy's matrix-like object.
        
        Params:
            data: a numpy's matrix-like object
        '''
        if isinstance(data, np.ndarray):
            self.mat = np.matrix(data)
        elif isinstance(data, np.matrix):
            self.mat = data
        elif (isinstance(data, Matrix)):
            warn("Automatic transform DenseMatrix to SparseMatrix")
            self.mat = data.toDenseMatrix().mat
        else:
            # TODO: raise suitable message
            raise TypeError("Only numpy matrix-like type allowed.")
        

    def toSparseMatrix(self):
        '''
        create a SparseMatrix from the matrix's data
        '''
        from composes.matrix.sparse_matrix import SparseMatrix
        return SparseMatrix(self.tocsr())
    
    def toDenseMatrix(self, copy = False):
        if (copy):
            return self.copy()
        else:
            return self    
        