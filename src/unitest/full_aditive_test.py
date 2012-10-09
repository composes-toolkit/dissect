'''
Created on Oct 8, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.composition.full_additive import FullAdditive
from composes.matrix.dense_matrix import DenseMatrix
from composes.utils.regression_learner import LstsqRegressionLearner

class Test(unittest.TestCase):


    def setUp(self):
        self.m11 = DenseMatrix(np.mat([[3],[9]]))
        self.m21 = DenseMatrix(np.mat([[4],[2]]))
        self.ph1 = DenseMatrix(np.mat([[18],[24]]))


    def tearDown(self):
        pass


    def test_train1(self):
        test_cases = [(self.m11, self.m21, self.ph1, np.mat([[2]]), np.mat([[3]])),
                      (self.m11, self.m21, DenseMatrix(np.mat([[0],[0]])),
                       np.mat([[0]]), np.mat([[0]]))]
        
        for m1, m2, ph, expected_a, expected_b in test_cases:
            comp_model = FullAdditive(learner=LstsqRegressionLearner(intercept=False))
            comp_model._train(m1, m2, ph)
            np.testing.assert_array_almost_equal(comp_model._mat_a_t.transpose().mat,
                                                 expected_a, 10)
            np.testing.assert_array_almost_equal(comp_model._mat_b_t.transpose().mat,
                                                 expected_b, 10)
     
     
    def test_train2(self):
        dim_ = 2
        dim_1 = 3
        dim_2 = 5        
        for dim in [dim_1 + dim_2, dim_1 + dim_2 + 2]:
            print dim    
            expected_a = np.mat(np.random.random((dim_,dim_1)))
            expected_b = np.mat(np.random.random((dim_,dim_2)))     
            m1 = np.mat(np.random.random((dim,dim_1)))     
            m2 = np.mat(np.random.random((dim,dim_2)))
           
            ph = np.mat(expected_a*m1.T + expected_b*m2.T)
           
            comp_model = FullAdditive(learner=LstsqRegressionLearner(intercept=False))
            comp_model._train(DenseMatrix(m1),DenseMatrix(m2),
                                       DenseMatrix(ph).transpose())
            np.testing.assert_array_almost_equal(comp_model._mat_a_t.transpose().mat,
                                                 expected_a, 10)
            np.testing.assert_array_almost_equal(comp_model._mat_b_t.transpose().mat,
                                                 expected_b, 10)
            
        for dim in [dim_1 + dim_2 + 6, dim_1 + dim_2 + 20]:
            print dim    
            expected_a = np.mat(np.random.random((dim_,dim_1)))
            expected_b = np.mat(np.random.random((dim_,dim_2)))     
            m1 = np.mat(np.random.random((dim,dim_1)))     
            m2 = np.mat(np.random.random((dim,dim_2)))
           
            ph = np.mat(expected_a*m1.T + expected_b*m2.T)
                       
            comp_model = FullAdditive(learner=LstsqRegressionLearner(intercept=True))
            comp_model._train(DenseMatrix(m1),DenseMatrix(m2),
                                       DenseMatrix(ph).transpose())
            np.testing.assert_array_almost_equal(comp_model._mat_a_t.transpose().mat,
                                                 expected_a, 10)
            np.testing.assert_array_almost_equal(comp_model._mat_b_t[:-1,:].transpose().mat,
                                                 expected_b, 10)
       
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()