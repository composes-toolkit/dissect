'''
Created on Oct 5, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.composition.weighted_additive import WeightedAdditive
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix

class Test(unittest.TestCase):


    def setUp(self):
        self.m11 = DenseMatrix(np.mat([[3],[9]]))
        self.m21 = DenseMatrix(np.mat([[4],[2]]))
        self.ph1 = DenseMatrix(np.mat([[18],[24]]))

        self.m12 = DenseMatrix(np.mat([[3,1],[9,2]]))
        self.m22 = DenseMatrix(np.mat([[4,3],[2,1]]))
        self.ph2 = DenseMatrix(np.mat([[18,11],[24,7]]))


    def tearDown(self):
        pass


    def test_train(self):
        test_cases = [(self.m11, self.m21, self.ph1, 2, 3),
                      (self.m12, self.m22, self.ph2, 2, 3),
                      (self.m11, self.m21, DenseMatrix(np.mat([[0],[0]])), 0, 0),
                      (SparseMatrix(self.m12), SparseMatrix(self.m22),
                        SparseMatrix(self.ph2), 2, 3),
                      (self.m11, DenseMatrix(np.mat([[0],[0]])), self.ph1, 3, 0),
                      (DenseMatrix(np.mat([[0],[0]])), self.m11, self.ph1, 0, 3),
                      (DenseMatrix(np.mat([[1,2,3]])),
                       DenseMatrix(np.mat([[2,4,6]])),
                       DenseMatrix(np.mat([[3,6,9]])), 0.6, 1.2),
                      (DenseMatrix(np.mat([[0],[0]])), 
                       DenseMatrix(np.mat([[0],[0]])), 
                       DenseMatrix(np.mat([[0],[0]])), 0.0, 0.0)
                      ]
        for m1, m2, ph, expected_alpha, expected_beta in test_cases:
            model = WeightedAdditive()
            model._train(m1, m2, ph)
            self.assertAlmostEqual(model.alpha, expected_alpha, 8) 
            self.assertAlmostEqual(model.beta, expected_beta, 8) 

    def test_compose(self):
        
        model = WeightedAdditive(2,3)
        np.testing.assert_array_equal(model._compose(self.m11, self.m21).mat,
                                                      self.ph1.mat)
        
        model = WeightedAdditive()
        np.testing.assert_array_equal(model._compose(self.m11, self.m21).mat,
                                                      np.mat([[7/2.],[11/2.]]))
        
        model = WeightedAdditive(0.5)
        np.testing.assert_array_equal(model._compose(self.m11, self.m21).mat,
                                                      np.mat([[7/2.],[11/2.]]))
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()