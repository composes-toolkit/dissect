'''
Created on Oct 1, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

import numpy as np
from dimensionality_reduction import DimensionalityReduction
from composes.matrix.linalg import Linalg
from math import sqrt

class Nmf(DimensionalityReduction):
    """
    Performs Non-negative Matrix Factorization to reduced dimension :math:`k`.
    
    Given an input non-negative matrix :math:`X`, it computes the decomposition:
    
    :math:`X \\approx WH` where W and H are non-negative matrices which minimize
    :math:`||X-WH||_{2}`

    It returns the matrix W.
    """
    
    _name = "nmf"

    def __init__(self, reduced_dimension):
        '''
        Constructor
        '''
        super(Nmf, self).__init__(reduced_dimension)
        
    def apply(self, matrix_):    
        
        matrix_.assert_positive()
        #w_init, h_init = self.nndsvd_init(matrix_)
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
            
            w[:, i] = (matrix_[:, rndcol].sum(1)/float(p_col)).flatten()
            h[i, :] = (matrix_[rndrow, :].sum(0)/float(p_row)).flatten()
            
        w = type(matrix_)(w)
        h = type(matrix_)(h)

        return w, h
    
    def nndsvd_init(self,matrix_):
        def matrix_abs(mat_):
            mat_p = mat_.get_non_negative()
            mat_n_abs = mat_p - mat_
            return mat_p + mat_n_abs
        
        def padd_zeros(matrix_, axis, thickness):
            matrix_type = type(matrix_)
            if axis == 0:  
                append_mat = matrix_type(np.zeros((thickness, matrix_.shape[1])))
                return matrix_.vstack(append_mat)
            elif axis == 1:
                append_mat = matrix_type(np.zeros((matrix_.shape[0], thickness)))
                return matrix_.hstack(append_mat)
        
        u, s, v = Linalg.svd(matrix_, self._reduced_dimension);
        
        rank = u.shape[1]
        w = [[]]*rank
        h = [[]]*rank
        
        vt = v.transpose()
        
        w[0] = sqrt(s[0]) * matrix_abs(u[:,0])
        h[0] = sqrt(s[0]) * matrix_abs(vt[0,:])
        
        for i in range(1,rank):
            uu = u[:,i]
            vv = vt[i,:]
            uup = uu.get_non_negative()
            uun = uup - uu
            vvp = vv.get_non_negative()
            vvn = vvp - vv
            
            n_uup = uup.norm()
            n_uun = uun.norm()
            n_vvp = vvp.norm()
            n_vvn = vvn.norm()
            
            termp = n_uup * n_vvp; termn = n_uun * n_vvn
            if (termp >= termn):
                w[i] = sqrt(s[i] * termp) * uup / n_uup 
                h[i] = sqrt(s[i] * termp) * vvp / n_vvp
            else:
                w[i] = sqrt(s[i] * termn) * uun / n_uun 
                h[i] = sqrt(s[i] * termn) * vvn / n_vvn
        
        w = matrix_.nary_hstack(w)
        h = matrix_.nary_vstack(h)
        
        w.remove_small_values(0.0000000001)
        h.remove_small_values(0.0000000001)
        
        if (rank < self._reduced_dimension):
            w = padd_zeros(w, 1, self._reduced_dimension - rank)
            h = padd_zeros(h, 0, self._reduced_dimension - rank)
        return w,h      
