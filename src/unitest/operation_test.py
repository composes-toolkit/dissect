'''
Created on Sep 26, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.weighting.ppmi import PpmiWeighting
from composes.dim_reduction.svd import Svd
from composes.dim_reduction.nmf import Nmf  
from composes.semantic_space.operation import WeightingOperation
from composes.semantic_space.operation import DimensionalityReductionOperation
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.exception.illegal_state_error import IllegalStateError


class Test(unittest.TestCase):


    def setUp(self):
        self.m1 = np.array([[1,2,3]])
        self.m2 = np.array([[3]])
        self.m3 = np.array([[4,2,6]])
        self.m4 = np.array([[2]])

        self.x = np.mat([[1,2,3],[2,4,6],[4,675,43]])
        self.us = np.mat([[  2.19272110e+00,   3.03174768e+00],
                               [  4.38544220e+00,   6.06349536e+00],
                               [  6.76369708e+02,  -4.91431927e-02]])
                
        self.xnmf = np.mat([[1,2,3],[2,4,6],[4,17,13]])
                              
    def tearDown(self):
        pass


    def test_apply_dimensionality_reduction(self):

        test_cases =[(self.x, self.us)] 
        red = Svd(2)
        
        for in_mat, expected_us_mat in test_cases:
            op = DimensionalityReductionOperation(red)
            tmp_mat = in_mat.copy()

            out_us_mat = op.apply(DenseMatrix(in_mat)).mat
            np.testing.assert_array_almost_equal(expected_us_mat, out_us_mat, 2)
            
            np.testing.assert_array_equal(in_mat, tmp_mat)
            self.assertRaises(IllegalStateError, op.apply, DenseMatrix(in_mat))
            self.assertRaises(IllegalStateError, op.apply, SparseMatrix(in_mat))
  
  
    def test_project_dimensionality_reduction(self):

        test_cases =[(self.x, self.us)] 
        red = Svd(2)
        
        for in_mat, expected_us_mat in test_cases:
            op = DimensionalityReductionOperation(red)
            tmp_mat = in_mat.copy()

            self.assertRaises(IllegalStateError, op.project, DenseMatrix(in_mat))
            
            op.apply(DenseMatrix(in_mat)).mat
            out_proj_mat = op.project(DenseMatrix(in_mat)).mat 
            np.testing.assert_array_almost_equal(expected_us_mat, out_proj_mat, 2)
            
            np.testing.assert_array_equal(in_mat, tmp_mat)
            
            self.assertRaises(IllegalStateError, op.apply, SparseMatrix(in_mat))
              
            out_proj_mat2 = op.project(DenseMatrix(in_mat)).mat 
            np.testing.assert_array_almost_equal(expected_us_mat, out_proj_mat2, 2)
            
    def test_project_dimensionality_reduction_nmf(self):

        test_cases = [self.xnmf] 
        red = Nmf(2)
        
        for in_mat in test_cases:
            d_in_mat = DenseMatrix(in_mat)
            op = DimensionalityReductionOperation(red)
            tmp_mat = in_mat.copy()

            self.assertRaises(IllegalStateError, op.project, d_in_mat)
            
            out_core_mat = op.apply(d_in_mat).mat
            out_proj_mat = op.project(d_in_mat).mat 
            np.testing.assert_array_almost_equal(out_proj_mat, out_core_mat, 5)
            
            np.testing.assert_array_equal(in_mat, tmp_mat)
            
            self.assertRaises(IllegalStateError, op.apply, d_in_mat)
              
            out_proj_mat2 = op.project(d_in_mat).mat 
            np.testing.assert_array_almost_equal(out_proj_mat2, out_core_mat, 5)
                    
        
    def test_apply_weighting_operation(self):
        test_cases = [(self.m1, np.array([[0,0,0]])),
                      (self.m2, np.array([[0]]))]
        w = PpmiWeighting()
        for in_mat, expected_mat in test_cases:
            op = WeightingOperation(w)
            tmp_mat = in_mat.copy()
            out_mat = op.apply(DenseMatrix(in_mat)).mat
            np.testing.assert_array_almost_equal(expected_mat, out_mat, 7)
            np.testing.assert_array_equal(in_mat, tmp_mat)
            self.assertRaises(IllegalStateError, op.apply, DenseMatrix(in_mat))
    
    def test_project_weighting_operation(self):
        test_cases = [(self.m1, self.m3, 
                       np.array([[0.69314718,0,0]])),
                      (self.m2, self.m4, np.array([[0]]))]
        w = PpmiWeighting()
        for (core_mat, per_mat, expected_mat) in test_cases:
            op = WeightingOperation(w)
            tmp_mat = per_mat.copy()
            
            self.assertRaises(IllegalStateError, op.project, 
                              DenseMatrix(per_mat))
            
            op.apply(DenseMatrix(core_mat))
            out_mat = op.project(DenseMatrix(per_mat)).mat
            np.testing.assert_array_almost_equal(expected_mat, out_mat, 7)
            np.testing.assert_array_equal(per_mat, tmp_mat)
            
            out_mat = op.project(DenseMatrix(per_mat)).mat
            np.testing.assert_array_almost_equal(expected_mat, out_mat, 7)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()