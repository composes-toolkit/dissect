'''
Created on Oct 15, 2012

@author: nghia
'''
import unittest
import numpy as np
from composes.matrix.dense_matrix import DenseMatrix
#from composes.composition.dilation_1 import DilationModel
from composes.composition.dilation import Dilation

class Test(unittest.TestCase):

    def setUp(self):
        self.m11 = DenseMatrix(np.mat([[4],[2]]))
        self.m21 = DenseMatrix(np.mat([[3],[6]]))
        self.ph1 = DenseMatrix(np.mat([[5],[10]]))
        
        self.m12 = DenseMatrix(np.mat([[2,0],[3,0]]))
        self.m22 = DenseMatrix(np.mat([[3,3],[6,4]]))
        self.ph2 = DenseMatrix(np.mat([[5,2],[10,5]]))
        
        self.m13 = DenseMatrix(np.mat([[4,3],[3,4]]))
        self.m23 = DenseMatrix(np.mat([[0,5],[0,5]]))
        self.ph3 = DenseMatrix(np.mat([[12,14],[12,21]]))
    
    def test_train_exact(self):
        test_cases = [(self.m11, self.m21, self.ph1, 5 / (3.0)),
                      (self.m12, self.m22, self.ph2, 5 / (3.0)),
                      (self.m13, self.m23, self.ph3, 6)]
        for arg1, arg2, phrase, lambda_ in test_cases:
            m = Dilation()
            m._train(arg1, arg2, phrase)
            self.assertAlmostEqual(m._lambda, lambda_)
    
    def test_train_random(self):
        scale_y = 1.5
        test_cases = [1.0,2.0,3.0]
        rows = 4
        cols = 3
        m1 = np.random.rand(rows,cols)
        m2 = np.random.rand(rows,cols)
        v_on_u = (m1 * m2).sum(axis=1) * (1.0 / (m1 * m1).sum(axis=1))
        x = m1 * v_on_u.reshape(rows,1)
        y = m2 - x
        for lambda_ in test_cases:
            m = Dilation()
            p = lambda_ * x + scale_y * y
            m._train(DenseMatrix(m1),DenseMatrix(m2),DenseMatrix(p))
            self.assertAlmostEqual(lambda_, m._lambda)
             
        
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()