'''
Created on Oct 4, 2012

@author: georgianadinu
'''
import unittest

import numpy as np
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.matrix.linalg import Linalg
from numpy.linalg.linalg import LinAlgError


class Test(unittest.TestCase):


    def test_dense_svd(self):
        test_cases = [(np.mat([[1,2,3],[2,4,6],[4,675,43]]), 
                       np.mat([[3.2418e-03,4.4720e-01],
                               [6.4836e-03,8.9440e-01],
                               [9.9997e-01,-7.2489e-03]]),
                       np.array([6.7639e+02,6.7794e+00]),
                       np.mat([[0.0059,0.9979,0.0636],
                               [0.3255,-0.0621,0.9434]]).transpose())
                        ]
        for x, u_expected, s_expected, v_expected in test_cases:
            u, s, v = Linalg.svd(DenseMatrix(x),2)
            np.testing.assert_array_almost_equal(u.mat, u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat, v_expected, 2)

            u, s, v = Linalg.svd(DenseMatrix(x),3)
            np.testing.assert_array_almost_equal(u.mat, u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat, v_expected, 2)
            
            u, s, v = Linalg.svd(DenseMatrix(x),6)
            np.testing.assert_array_almost_equal(u.mat, u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat, v_expected, 2)
            
            u, s, v = Linalg.svd(DenseMatrix(x),1)
            np.testing.assert_array_almost_equal(u.mat, u_expected[:,0:1], 2)
            np.testing.assert_array_almost_equal(s, s_expected[0:1], 2)
            np.testing.assert_array_almost_equal(v.mat, v_expected[:, 0:1], 2)
    
    def test_sparse_svd(self):
        test_cases = [(np.mat([[1,2,3],[2,4,6],[4,675,43]]), 
                       np.mat([[3.2418e-03,4.4720e-01],
                               [6.4836e-03,8.9440e-01],
                               [9.9997e-01,-7.2489e-03]]),
                       np.array([6.7639e+02,6.7794e+00]),
                       np.mat([[0.0059,0.9979,0.0636],
                               [0.3255,-0.0621,0.9434]]).transpose())
                        ]
        for x, u_expected, s_expected, v_expected in test_cases:
            u, s, v = Linalg.svd(SparseMatrix(x),2)
            np.testing.assert_array_almost_equal(u.mat.todense(), u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat.todense(), v_expected, 2)

            u, s, v = Linalg.svd(SparseMatrix(x),3)
            np.testing.assert_array_almost_equal(u.mat.todense(), u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat.todense(), v_expected, 2)
            
            u, s, v = Linalg.svd(SparseMatrix(x),6)
            np.testing.assert_array_almost_equal(u.mat.todense(), u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat.todense(), v_expected, 2)
            
            u, s, v = Linalg.svd(SparseMatrix(x),1)
            np.testing.assert_array_almost_equal(u.mat.todense(), u_expected[:,0:1], 2)
            np.testing.assert_array_almost_equal(s, s_expected[0:1], 2)
            np.testing.assert_array_almost_equal(v.mat.todense(), v_expected[:,0:1], 2)
    
    def test_svd_raises(self):
        test_cases = [np.mat([[1,2,3],[2,4,6],[4,675,43]])]
        
        for x in test_cases:   
            self.assertRaises(ValueError, Linalg.svd, DenseMatrix(x), 0)
            self.assertRaises(ValueError, Linalg.svd, SparseMatrix(x), 0)
    
    def test_pinv(self):
        test_cases = [(np.mat([[1,0,0],[0,1,0],[0,-4,1]]),
                       np.mat([[1,0,0],[0,1,0],[0,4,1]])),
                      (np.mat([[3,4],[6,8]]),
                       np.mat([[0.024,0.048],[0.032,0.064]])),
                      (np.mat([[1,1,1],[1,1,0],[1,1,1]]),
                       np.mat([[1.11022302e-16,5.00000000e-01,5.55111512e-17],
                               [-5.55111512e-17,5.00000000e-01,-1.11022302e-16],
                               [5.00000000e-01,-1.00000000e+00,5.00000000e-01]]))]
         
        for in_mat, expected_out in test_cases:
            out_mat = Linalg.pinv(DenseMatrix(in_mat))
            np.testing.assert_array_almost_equal(out_mat.mat, expected_out, 7)
            
            out_mat = Linalg.pinv(SparseMatrix(in_mat))
            np.testing.assert_array_almost_equal(out_mat.mat.todense(), 
                                                 expected_out, 7)
          
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()