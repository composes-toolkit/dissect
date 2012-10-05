'''
Created on Oct 5, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.feature_selection.top_feature_selection import TopFeatureSelection
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix

class Test(unittest.TestCase):

    def setUp(self):
        self.a = np.array([[1,2,3],[4,0,5]]) 
    

    def test_top_feat_selection(self):
        test_cases = [(self.a, np.mat([[3,1],[5,4]]), [2,0], 2), 
                      (self.a, np.mat([[3],[5]]), [2], 1),
                      (self.a, np.mat([[3,1,2],[5,4,0]]), [2,0, 1], 6),
                      ]
        
        for in_mat, expected_mat, expected_perm, no_cols in test_cases:
            fs = TopFeatureSelection(no_cols)
            
            out_mat, perm = fs.apply(DenseMatrix(in_mat))
            np.testing.assert_array_equal(out_mat.mat, expected_mat)
            self.assertListEqual(perm, expected_perm)
            
            out_mat, perm = fs.apply(SparseMatrix(in_mat))
            np.testing.assert_array_equal(out_mat.mat.todense(), expected_mat)
            self.assertListEqual(perm, expected_perm)
            
            fs = TopFeatureSelection(no_cols, criterion="length")
            
            out_mat, perm = fs.apply(DenseMatrix(in_mat))
            np.testing.assert_array_equal(out_mat.mat, expected_mat)
            self.assertListEqual(perm, expected_perm)
            
            out_mat, perm = fs.apply(SparseMatrix(in_mat))
            np.testing.assert_array_equal(out_mat.mat.todense(), expected_mat)
            self.assertListEqual(perm, expected_perm)
      
        self.assertRaises(ValueError, TopFeatureSelection, 0)
        self.assertRaises(ValueError, TopFeatureSelection, 2, criterion="something")

                        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_top_feat_selection']
    unittest.main()