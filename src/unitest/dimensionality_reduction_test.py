'''
Created on Sep 28, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import unittest
import numpy as np
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.dim_reduction.nmf import Nmf
from composes.matrix.linalg import Linalg
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix

class DimReductionTest(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

    def test_nmf(self):
        test_cases = [np.mat([[1,2,3],[2,4,6],[4,17,13]], dtype = np.double),
                      np.mat([[1,0,0]], dtype = np.double)]

        for in_mat in test_cases:
            red = Nmf(2)
            d_mat = DenseMatrix(in_mat)
            #wd_init, hd_init = red.random_init(d_mat)
            wd_init, hd_init = red.v_col_init(d_mat)

            s_mat = SparseMatrix(in_mat)
            ws_init = SparseMatrix(wd_init)
            hs_init = SparseMatrix(hd_init)

            wd_mat, hd_mat = Linalg.nmf(d_mat, wd_init, hd_init)
            ws_mat, hs_mat = Linalg.nmf(s_mat, ws_init, hs_init)

            #TESTED IT AGAINST MATLAB IMPLEMENTATION - ALL GOOD
            #print wd_mat.mat
            #print hd_mat.mat
            #print ws_mat.mat.todense()
            #print hs_mat.mat.todense()
            print("V:", in_mat)
            print("WH:", (ws_mat*hs_mat).mat.todense())

            np.testing.assert_array_almost_equal(wd_mat.mat,
                                                 ws_mat.mat.todense(), 2)
            np.testing.assert_array_almost_equal(hd_mat.mat,
                                                 hs_mat.mat.todense(), 2)

    def test_svd(self):
        test_cases = [(DenseMatrix(np.mat([[1,2,3],[2,4,6],[4,675,43]])),
                       np.mat([[  2.19272110e+00,   3.03174768e+00, 0],
                               [  4.38544220e+00,   6.06349536e+00, 0],
                               [  6.76369708e+02,  -4.91431927e-02, 0]]),
                       np.mat([[0.0059,0.9979,0.0636],
                               [0.3255,-0.0621,0.9434],
                               [0.945,0.015,-0.325]]).transpose())]



        for x, us_expected, v_expected in test_cases:

            svd_red = Svd(2)
            us, transmat = svd_red.apply(x)
            np.testing.assert_array_almost_equal(us.mat, us_expected[:,0:2], 2)
            np.testing.assert_array_almost_equal(transmat.mat, v_expected[:,0:2], 2)

            svd_red = Svd(3)
            us, transmat = svd_red.apply(x)
            np.testing.assert_array_almost_equal(us.mat, us_expected[:,0:2], 2)
            np.testing.assert_array_almost_equal(transmat.mat, v_expected[:,0:2], 2)

            svd_red = Svd(6)
            us, transmat = svd_red.apply(x)
            np.testing.assert_array_almost_equal(us.mat, us_expected[:,0:2], 2)
            np.testing.assert_array_almost_equal(transmat.mat, v_expected[:,0:2], 2)

            svd_red = Svd(1)
            us, transmat = svd_red.apply(x)
            np.testing.assert_array_almost_equal(us.mat, us_expected[:,0:1], 2)
            np.testing.assert_array_almost_equal(transmat.mat, v_expected[:,0:1], 2)


        test_cases = [(SparseMatrix(np.mat([[1,2,3],[2,4,6],[4,675,43]])),
                       np.mat([[  2.19272110e+00,   3.03174768e+00, 0],
                               [  4.38544220e+00,   6.06349536e+00, 0],
                               [  6.76369708e+02,  -4.91431927e-02, 0]]),
                       np.mat([[0.0059,0.9979,0.0636],
                               [0.3255,-0.0621,0.9434],
                               [0.945,0.015,-0.325]]).transpose())]


        for x, us_expected, v_expected in test_cases:
            us_expected = np.abs(us_expected)
            v_expected = np.abs(v_expected)

            svd_red = Svd(2)
            us, transmat = svd_red.apply(x)
            np.testing.assert_array_almost_equal(np.abs(us.mat.todense()), us_expected[:,0:2], 2)
            np.testing.assert_array_almost_equal(np.abs(transmat.mat.todense()), v_expected[:,0:2], 2)

            svd_red = Svd(3)
            us, transmat = svd_red.apply(x)
            np.testing.assert_array_almost_equal(np.abs(us.mat.todense()), us_expected[:,0:2], 2)
            np.testing.assert_array_almost_equal(np.abs(transmat.mat.todense()), v_expected[:,0:2], 2)

            svd_red = Svd(6)
            us, transmat = svd_red.apply(x)
            np.testing.assert_array_almost_equal(np.abs(us.mat.todense()), us_expected[:,0:2], 2)
            np.testing.assert_array_almost_equal(np.abs(transmat.mat.todense()), v_expected[:,0:2], 2)

            svd_red = Svd(1)
            us, transmat = svd_red.apply(x)
            np.testing.assert_array_almost_equal(np.abs(us.mat.todense()), us_expected[:,0:1], 2)
            np.testing.assert_array_almost_equal(np.abs(transmat.mat.todense()), v_expected[:,0:1], 2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_svd']
    unittest.main()