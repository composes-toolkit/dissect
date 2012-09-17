'''
Created on Sep 17, 2012

@author: georgianadinu
'''

import numpy as np
from warnings import warn
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from composes.matrix.matrix import Matrix



class SparseMatrix(object):
    '''
    classdocs
    '''


    def __init__(self, data, *args, **kwargs):
        '''
        Constructor, create a SparseMatrix from a scipy sparse matrix.
        
        Params:
            data<sparse matrix>: a scipy sparse matrix
        '''
        if (isinstance(data, csr_matrix)):
            '''
            If the input data is a csr_matrix, use it without copying
            '''
            self.mat = data
        elif (issparse(data)):
            '''
            If the input data is a sparse matrix, convert it to csr_matrix
            '''
            self.mat = data.tocsr()
        elif (isinstance(data, np.matrix) or isinstance(data, np.ndarray)):
            '''
            if the input data is a numpy.matrix, convert it to csr_matrix
            '''
            warn("Automatic transform numpy's dense matrix to scipy's sparse matrix")
            self.mat = csr_matrix(data)
        elif (isinstance(data, Matrix)):
            warn("Automatic transform DenseMatrix to SparseMatrix")
            self.mat = data.toSparseMatrix().mat
        else:
            raise TypeError("data should be a scipy sparse matrix instead of" + str(type(data)))
        

    def toDenseMatrix(self):
        '''
        Create a DenseMatrix from the matrix's data.
        '''
        from composes.matrix.dense_matrix import DenseMatrix
        return DenseMatrix(self.todense())
    
    def toSparseMatrix(self, copy = False):
        if (copy):
            return self.copy()
        else:
            return self    
    