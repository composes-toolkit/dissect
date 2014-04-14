'''
Created on Oct 12, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import unittest
import numpy as np
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.utils.matrix_utils import resolve_type_conflict
from scipy.sparse import csr_matrix

class Test(unittest.TestCase):


    def test_resolve_type_conflict(self):

        arr = np.mat([1,2])

        a = DenseMatrix(arr)
        b = SparseMatrix(arr)

        [c,d] = resolve_type_conflict([a,b], DenseMatrix)
        [e,f,g] = resolve_type_conflict([b,a,a], DenseMatrix)
        h = resolve_type_conflict([], DenseMatrix)

        [u,v] = resolve_type_conflict([arr, csr_matrix(arr)], DenseMatrix)

        self.assertIsInstance(c, DenseMatrix)
        self.assertIsInstance(d, DenseMatrix)
        self.assertIsInstance(e, DenseMatrix)
        self.assertIsInstance(f, DenseMatrix)
        self.assertIsInstance(g, DenseMatrix)
        self.assertListEqual([], h)

        self.assertIsInstance(g, DenseMatrix)

        self.assertIsInstance(u, DenseMatrix)
        self.assertIsInstance(v, DenseMatrix)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()