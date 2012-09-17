'''
Created on Sep 17, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
import numpy.testing 
from scipy.sparse import csr_matrix 
from composes.matrix.sparse_matrix import SparseMatrix
from composes.matrix.dense_matrix import DenseMatrix
    
class TestDenseMatrix(unittest.TestCase):
    
    
    def setUp(self):
        self.a = np.array([[1,2,3],[4,0,5]])
        self.m = DenseMatrix(self.a)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertIsInstance(self.m.mat, np.matrix)
        numpy.testing.assert_array_equal(self.a, np.array(self.m.mat))

    def test_add(self):
        pass


class TestSparseMatrix(unittest.TestCase):
    
    def setUp(self):
        self.a = np.array([[1,2,3],[4,0,5]])
        self.b = np.array([[0,0,0],[0,0,0]])
        self.m = SparseMatrix(self.b)
        
    def tearDown(self):
        pass

    def test_init(self):
        self.assertIsInstance(self.m.mat, csr_matrix)
        
        numpy.testing.assert_array_equal(self.b, np.array(self.m.mat.todense()))
        
    def test_add(self):
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()        