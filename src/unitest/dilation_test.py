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
        #self.ph1 = DenseMatrix(np.mat([[5],[10]]))
        self.ph1 = DenseMatrix(np.mat([[80],[40]]))
        
        self.m12 = DenseMatrix(np.mat([[2,0],[3,0]]))
        self.m22 = DenseMatrix(np.mat([[3,3],[6,4]]))
        #self.ph2 = DenseMatrix(np.mat([[5,2],[10,5]]))
        self.ph2 = DenseMatrix(np.mat([[20,8],[90,45]]))
        
        self.m13 = DenseMatrix(np.mat([[4,3],[3,4]]))
        self.m23 = DenseMatrix(np.mat([[0,5],[0,5]]))
        #self.ph3 = DenseMatrix(np.mat([[12,14],[12,21]]))
        self.ph3 = DenseMatrix(np.mat([[300,350],[300,525]]))
        
        self.m14 = DenseMatrix(np.mat([[4,3],[3,4],[0,0]]))
        self.m24 = DenseMatrix(np.mat([[0,5],[0,5],[0,0]]))
        #self.ph4 = DenseMatrix(np.mat([[12,14],[12,21],[0,0]]))
        self.ph4 = DenseMatrix(np.mat([[300,350],[300,525],[0,0]]))
        
        self.m15 = DenseMatrix(np.mat([[2,0],[0,0],[3,0]]))
        self.m25 = DenseMatrix(np.mat([[3,3],[0,0],[6,4]]))
        #self.ph5 = DenseMatrix(np.mat([[5,2],[0,0],[10,5]]))
        self.ph5 = DenseMatrix(np.mat([[20,8],[0,0],[90,45]]))
        
        self.m16 = DenseMatrix(np.mat([[0,0],[0,0]]))
        self.m26 = DenseMatrix(np.mat([[0,0],[0,0]]))
        self.ph6 = DenseMatrix(np.mat([[0,0],[0,0]]))
        
        self.m17 = DenseMatrix(np.mat([[2,0],[3,0]]))
        self.m27 = DenseMatrix(np.mat([[0,1],[0,2]]))
        #self.ph7 = DenseMatrix(np.mat([[4,5],[5,4]]))
        self.ph7 = DenseMatrix(np.mat([[16,20],[45,36]]))
    
    def test_train_exact(self):
        test_cases = [(self.m11, self.m21, self.ph1, 5 / (3.0)),
                      (self.m12, self.m22, self.ph2, 5 / (3.0)),
                      (self.m13, self.m23, self.ph3, 6),
                      (self.m14, self.m24, self.ph4, 6),
                      (self.m15, self.m25, self.ph5, 5 / (3.0)),
                      (self.m16, self.m26, self.ph6, 2),
                      (self.m17, self.m27, self.ph7, 2)
                      ]
        
        for arg1, arg2, phrase, lambda_ in test_cases:
            m = Dilation()
            m._train(arg1, arg2, phrase)
            self.assertAlmostEqual(m._lambda, lambda_)
#            
    def test_compose_exact(self):
        
        test_cases = [(self.m11, self.m21, self.ph1, 5 / (3.0)),
                      (self.m13, self.m23, self.ph3, 6),
                      (self.m14, self.m24, self.ph4, 6)
                      ]
        for arg1, arg2, phrase, lambda_ in test_cases:
            
            m = Dilation()
            m._train(arg1, arg2, phrase)
            res = m._compose(arg1, arg2)
            np.testing.assert_array_almost_equal(res.mat, phrase.mat, 2)
            
            m = Dilation(lambda_)
            res = m._compose(arg1, arg2)
            np.testing.assert_array_almost_equal(res.mat, phrase.mat, 2)
            
        
    def test_train_random(self):
        test_cases = [1.0,2.0,3.0]
        rows = 4
        cols = 3
        m1 = np.random.rand(rows,cols)
        m2 = np.random.rand(rows,cols)
        
        
        for lambda_ in test_cases:
            m = Dilation(lambda_)
            result_p = m._compose(DenseMatrix(m1), DenseMatrix(m2))
            
            m = Dilation()
            m._train(DenseMatrix(m1),DenseMatrix(m2),result_p)
            self.assertAlmostEqual(lambda_, m._lambda)
 
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()