'''
Created on Oct 1, 2012

@author: georgianadinu
'''

import numpy as np
from dimensionality_reduction import DimensionalityReduction
from composes.matrix.linalg import Linalg
from math import sqrt

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
    
    def nndsvd_init(self, matrix):
        pass
    def nndsvd_init_tmp(self,matrix_, reduced_dimension, flag=0):
        def matrix_abs(mat_):
            mat_p = mat_.get_non_negative()
            mat_n_abs = mat_p - mat_
            return mat_p + mat_n_abs 
        
        matrix_.assert_positive()
        w = [[]]*reduced_dimension
        h = [[]]*reduced_dimension
        u, s, vt = Linalg.svd(matrix_, reduced_dimension);
        print "reduced dimension", u.get_shape()[1]
        
        w[0] = sqrt(s[0]) * matrix_abs(u[:,0])
        h[0] = sqrt(s[0]) * matrix_abs(vt[0,:])
        
        for i in range(1,reduced_dimension):
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
        
        # TODO: implement function in matrices
        w.removeNearZeros(0.0000000001)
        h.removeNearZeros(0.0000000001)
        
        if( flag==1 ): #NNDSVDa: fill in the zero elements with the average        
            # TODO: implement later
            raise NotImplementedError()
            #ind1 = W==0
            #ind2 = H==0
            #average = Matrices.mean(a) 
            #W[ind1] = average
            #H[ind2] = average;
        
        return w,h      
