'''
Created on Sep 26, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.weighting.ppmi import PpmiWeighting
from composes.operation import WeightingOperation
from composes.matrix.dense_matrix import DenseMatrix
from composes.exception.illegal_state_error import IllegalStateError


class Test(unittest.TestCase):


    def setUp(self):
        self.m1 = np.array([[1,2,3]])
        self.m2 = np.array([[3]])
        self.m3 = np.array([[4,2,6]])
        self.m4 = np.array([[2]])


    def tearDown(self):
        pass


    def test_apply_weighting_operation(self):
        test_cases = [(self.m1, np.array([[0,0,0]])),
                      (self.m2, np.array([[0]]))]
        w = PpmiWeighting()
        for (in_mat, expected_mat) in test_cases:
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