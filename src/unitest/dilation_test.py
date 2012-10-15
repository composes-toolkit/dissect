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
    
    def testTrain(self):
        test_cases = [(self.m11, self.m21, self.ph1, 5 / (3.0))]
        for arg1, arg2, phrase, lambda_ in test_cases:
            m = Dilation()
            m._train(arg1, arg2, phrase)
            self.assertAlmostEqual(m._lambda, lambda_)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()