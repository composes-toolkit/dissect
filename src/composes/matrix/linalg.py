'''
Created on Oct 4, 2012

@author: georgianadinu
'''

import numpy as np
import logging
import scipy.linalg as splinalg
from sparsesvd import sparsesvd
from warnings import warn
from time import time
from math import sqrt
from composes.matrix.matrix import Matrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.utils.matrix_utils import assert_same_shape
from composes.utils.matrix_utils import padd_matrix
import composes.utils.log_utils as log

logger = logging.getLogger(__name__)

class Linalg(object):
    """
    Contains a set of liniar algebra utilities defined to work both with sparse and
    with dense matrices as an input (i.e. with objects of type SparseMatrix/DenseMatrix).
    
    Implements:
        svd
        nmf
        pinv
    """

    _NMF_ALPHA = 1.0
    _NMF_BETA = 0.1
    _NMF_MAX_ITER = 20
    _NMF_MAX_ITER_SUBPROB = 10 
    _NMF_MIN_TOL = 0.001
    _NMF_TOL = _NMF_MIN_TOL
    _NMF_TOL_DECREASE_FACTOR = 0.5
    _NMF_TIME_LIMIT = 7000

    @staticmethod
    def svd(matrix_, reduced_dimension):
        """
        Performs SVD decomposition.
        
        If the rank is smaller than the requested reduced dimension,
        reduction to rank is performed.
        
        Args:
           matrix_: input of type Matrix
           reduced_dimension: int, the desired reduced dimension
        
        Returns:
            U,S,V of the decomposition X = USV^T. U, V: Matrix type, 
            S: ndarray of singular values.
            
        """
        log.print_info(logger, 4, "In SVD..reducing to dim %d" % reduced_dimension)
        log.print_matrix_info(logger, matrix_, 5, "Input matrix:")

        #TODO: IMPORTANT!! do the sign normalization COLUMN-wise!!!not 
        #for the full matrix at once!!
        if reduced_dimension == 0:
            raise ValueError("Cannot reduce to dimensionality 0.")
        
        if isinstance(matrix_, SparseMatrix):
            result =  Linalg._sparse_svd(matrix_, reduced_dimension)
        elif isinstance(matrix_, DenseMatrix):
            result =  Linalg._dense_svd(matrix_, reduced_dimension)
        else:
            raise TypeError("expected Matrix type, received %s" % type(matrix_))
        
        log.print_matrix_info(logger, result[0], 5, "Resulted matrix U:")
        return result
        
    @staticmethod
    def ridge_regression(matrix_a , matrix_b, lambda_, intercept=False):
        #log.print_info(logger, "In Ridge regression..", 4)
        #log.print_matrix_info(logger, matrix_a, 5, "Input matrix A:")
        #log.print_matrix_info(logger, matrix_b, 5, "Input matrix B:")   
        """
        Performs Ridge Regression.
        
        This method use the general formulae:
            X = (A^T * A + P^T * P)^-1 * A^T * Y 
        to solve simple ridge regression
        
        Args:
            a_matrix <Matrix>: A
            b_matrix <Matrix>: B
            lambda_: scalar, lambda parameter
            intercept: bool. If True intercept is used. Optional, False by default.
                 
        Returns:
            X <Matrix>: X matrix, A * X = B
                        
        """
        
        matrix_a._assert_same_type(matrix_b)
        # TODO: check out where to define this assert
        assert_same_shape(matrix_a, matrix_b, 0)
        
        matrix_type = type(matrix_a)
        dim = matrix_a.shape[1]
        
        if intercept:
            matrix_a = matrix_a.hstack(matrix_type(np.ones((matrix_a.shape[0],
                                                             1))))
        lambda_diag = (lambda_ * lambda_) * matrix_type.identity(dim)
        
        if intercept:
            lambda_diag = padd_matrix(padd_matrix(lambda_diag, 0, 0.0), 1, 0.0)
        
        tmp_mat = Linalg.pinv(((matrix_a.transpose() * matrix_a) + lambda_diag))
        
        result = (tmp_mat * matrix_a.transpose()) * matrix_b
        
        return result
    
    @classmethod
    def lstsq_regression(cls, matrix_a, matrix_b, intercept=False):
        """
        Performs Least Squares Regression.
        
        Args:
            a_matrix <Matrix>: A
            b_matrix <Matrix>: B
            intercept: bool. If True intercept is used. Optional, False by default.
                 
        Returns:
            X <Matrix>: X matrix, A * X = B
                        
        """

        matrix_a._assert_same_type(matrix_b)
        # TODO: check out where to define this assert
        assert_same_shape(matrix_a, matrix_b, 0)

        if intercept:
            matrix_a = matrix_a.hstack(type(matrix_a)(np.ones((matrix_a.shape[0],
                                                             1))))
        if isinstance(matrix_a, DenseMatrix):
            result = Linalg._dense_lstsq_regression(matrix_a, matrix_b)
        else:
            result = Linalg._sparse_lstsq_regression(matrix_a, matrix_b)
        
        return result

    @staticmethod
    def _dense_lstsq_regression(matrix_a , matrix_b):
        return DenseMatrix(Linalg._numpy_lstsq_regression(matrix_a, matrix_b))
        #return DenseMatrix(Linalg._scipy_lstsq_regression(matrix_a, matrix_b))
    
    @staticmethod
    def _sparse_lstsq_regression(matrix_a , matrix_b, intercept=False):
        return Linalg.ridge_regression(matrix_a, matrix_b, 0.0)
        #return SparseMatrix(Linalg._dense_lstsq_regression(DenseMatrix(matrix_a), 
        #                                      DenseMatrix(matrix_b)))
        
    @staticmethod
    def _numpy_lstsq_regression(matrix_a, matrix_b, rcond=-1):
        return np.linalg.lstsq(matrix_a.mat, matrix_b.mat, rcond)[0]
    
    @staticmethod
    def _scipy_lstsq_regression(matrix_a, matrix_b):
        return splinalg.lstsq(matrix_a.mat, matrix_b.mat)[0]
    
    @staticmethod
    def _sparse_svd(matrix_, reduced_dimension):
                #svds from scipy.sparse.linalg
        #RAISES ValueError if the rank is smaller than reduced_dimension + 1
        #TODO : fix this or replace with svdsparse
        #??? eIGENVALUES ARE NOT SORTED!!!!!!
        #IF EVER USE THIS; FIX THE PROBLEMS
        #u, s, vt = svds(matrix_.mat, False, True)
        
        ut, s, vt = sparsesvd(matrix_.mat.tocsc(), reduced_dimension)
        rank = ut.shape[0]
        u = SparseMatrix(ut.transpose())
        v = SparseMatrix(vt.transpose())
        
        Linalg._check_svd_rank(matrix_.shape[1], rank, reduced_dimension) 
        
        if not u.is_mostly_positive():
            u = -u
            v = -v
            
        return u, s, v
    
    @staticmethod
    def _dense_svd(matrix_, reduced_dimension):
        '''
           - return three outputs
            + u: u matrix
            + s: flat version of s matrix
            + vt: transpose of v matrix
        '''
        u, s, vt = np.linalg.svd(matrix_.mat, False, True)
        tol = 1e-12
        rank = len(s[s > tol])
        
        Linalg._check_svd_rank(matrix_.shape[1], rank, reduced_dimension)
                            
        no_cols = min(rank, reduced_dimension)
        u = DenseMatrix(u[:,0:no_cols])
        s = s[0:no_cols]
        v = DenseMatrix(vt[0:no_cols,:].transpose())
        
        if not u.is_mostly_positive():
            u = -u
            v = -v

        return u, s, v
    
    @staticmethod
    def _check_svd_rank(no_columns, rank, reduced_dimension):
        if reduced_dimension > no_columns:
            warn("Number of columns smaller than the reduced dimensionality requested: %d < %d. Truncating to %d dimensions (rank)." % (no_columns, reduced_dimension, rank))
        elif reduced_dimension > rank:
            warn("Rank of matrix smaller than the reduced dimensionality requested: %d < %d. Truncating to %d dimensions." % (rank, reduced_dimension, rank))
    
    @staticmethod
    def _nmf_nlssubprob(v, w, w_t, h_init, tol, maxiter):
        """
        h, grad: output solution and gradient
        iteration: #iterations used
        v, w: constant matrices
        h_init: initial solution
        tol: stopping tolerance
        maxiter: limit of iterations
        """
        h = h_init
        w_t_v = w_t * v
        w_t_w = w_t * w 

        alpha = Linalg._NMF_ALPHA
        beta = Linalg._NMF_BETA
        
        #sub_loop_time = time()
        
        for iteration in xrange(1, maxiter):
            grad = w_t_w * h - w_t_v

            # search step size 
            for inner_iter in xrange(1, 20):
                hn = h - alpha * grad
                hn = hn.get_non_negative()
                d = hn - h
                gradd = grad.multiply(d).sum()
                dQd = (w_t_w * d).multiply(d).sum()
                suff_decr = 0.99 * gradd + 0.5 * dQd < 0
                if inner_iter == 1:
                    decr_alpha = not suff_decr
                    hp = h
                if decr_alpha: 
                    if suff_decr:
                        h = hn
                        break
                    else:
                        alpha = alpha * beta
                else:
                    if not suff_decr or hp.all_close(hn):
                        h = hp
                        break
                    else:
                        alpha = alpha / beta
                        hp = hn

        return h, grad, iteration

    @staticmethod
    def nmf(v, w_init, h_init):
        """
        Performs Non-negative Matrix Factorization.
        
        Args:
            w_init <Matrix>: initial value for matrix W
            h_init <Matrix>: initial value for matrix H
                 
        Returns:
            W, H <Matrix>: where X = WH
                        
        """
        
        log.print_info(logger, 4, "In NMF..reducing to dim %d" % w_init.shape[1])
        log.print_matrix_info(logger, w_init, 5, "W init matrix:")
        log.print_matrix_info(logger, h_init, 5, "H init matrix:")

        if not isinstance(v, Matrix):
            raise TypeError("expected Matrix type, received %s" % type(v))
        w = w_init
        h = h_init
        init_time = time()
        
        wt = w.transpose()
        ht = h.transpose()
        vt = v.transpose()
        gradW = (w * (h * ht)) - (v * ht)
        gradH = ((wt * w) * h) - (wt * v)

        gradW_norm = gradW.norm()
        gradH_norm = gradH.norm()
        initgrad = sqrt(pow(gradW_norm, 2) + pow(gradH_norm, 2))
        
        #print 'Init gradient norm %f' % initgrad 
        tolW = max(Linalg._NMF_MIN_TOL, Linalg._NMF_TOL) * initgrad
        tolH = tolW
        
        #loop_time = init_time
        for iteration in xrange(1, Linalg._NMF_MAX_ITER):
            log.print_info(logger, 5, "Iteration: %d(%d)" % (iteration, Linalg._NMF_MAX_ITER))
          
            if time() - init_time > Linalg._NMF_TIME_LIMIT:
                break
          
            w, gradW, iterW = Linalg._nmf_nlssubprob(vt, h.transpose(), h,
                                              w.transpose(), tolW, 
                                              Linalg._NMF_MAX_ITER_SUBPROB)
            old_w = w
            w = w.transpose()
            gradW = gradW.transpose()
          
            if iterW == 1:
                tolW = Linalg._NMF_TOL_DECREASE_FACTOR * tolW
        
            h, gradH, iterH = Linalg._nmf_nlssubprob(v, w, old_w, h, tolH, 
                                              Linalg._NMF_MAX_ITER_SUBPROB)
            
            if iterH == 1:
                tolH = Linalg._NMF_TOL_DECREASE_FACTOR * tolH
        
        log.print_matrix_info(logger, w, 5, "Return W matrix:")
        log.print_matrix_info(logger, h, 5, "Return H matrix:")
        return w, h
    
    @staticmethod
    def pinv(matrix_):
        if isinstance(matrix_, SparseMatrix):
            return Linalg._sparse_pinv(matrix_)
        elif isinstance(matrix_, DenseMatrix):
            return Linalg._dense_pinv(matrix_)
        else:
            raise TypeError("expected Matrix type, received %s" % type(matrix_))
    
    @staticmethod
    def _dense_pinv(matrix_):
        return DenseMatrix(np.linalg.pinv(matrix_.mat))
    
    @staticmethod
    def _sparse_pinv(matrix_):
        # TODO: implement pinv
        return SparseMatrix(np.linalg.pinv(matrix_.mat.todense()))
    