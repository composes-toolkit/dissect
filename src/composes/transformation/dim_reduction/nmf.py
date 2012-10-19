'''
Created on Oct 1, 2012

@author: georgianadinu
'''

import numpy as np
from dimensionality_reduction import DimensionalityReduction
from composes.matrix.linalg import Linalg

class Nmf(DimensionalityReduction):
    '''
    classdocs
    '''
    _name = "nmf"

    def __init__(self, reduced_dimension):
        '''
        Constructor
        '''
        super(Nmf, self).__init__(reduced_dimension)
        
    def apply(self, matrix_):    
        
        matrix_.assert_positive()
        w_init, h_init = self.v_col_init(matrix_)
        #w_init, h_init = self.random_init(matrix_)
        w, h = Linalg.nmf(matrix_, w_init, h_init) 
        return w, Linalg.pinv(h)
        
    def random_init(self, matrix_):
        
        # TODO: implement the fancier but still fast init (from nimfa: v_col)  
        rndcol = np.random.random_integers(0, matrix_.shape[1] - 1, 
                                           self._reduced_dimension)
        
        rndrow = np.random.random_integers(0, matrix_.shape[0] - 1, 
                                           self._reduced_dimension)
        
        #otherwise we would have had to convert to DenseMatrix/SparseMatrix
        #type(matrix_)(result)
        w = matrix_[:, rndcol]
        h = matrix_[rndrow, :]
        
        return w, h
    
    def v_col_init(self, matrix_):
        w = np.zeros((matrix_.shape[0], self._reduced_dimension))
        h = np.zeros((self._reduced_dimension, matrix_.shape[1]))

        #in case there are less than 5 rows or columns
        p_col = matrix_.shape[1]//5 + 1
        p_row = matrix_.shape[0]//5 + 1
        for i in range(self._reduced_dimension):
            
            rndcol = np.random.random_integers(0, matrix_.shape[1] - 1, 
                                           p_col)
        
            rndrow = np.random.random_integers(0, matrix_.shape[0] - 1, 
                                           p_row)
        
            w[:, i] = matrix_[:, rndcol].sum(0)/float(p_col)
            h[i, :] = matrix_[rndrow, :].sum(1)/float(p_row)
            
        w = type(matrix_)(w)
        h = type(matrix_)(h)

        return w, h
                
    