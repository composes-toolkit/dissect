'''
Created on Oct 2, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import unittest
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.similarity.dot_prod import DotProdSimilarity
from composes.similarity.cos import CosSimilarity
from composes.similarity.euclidean import EuclideanSimilarity
from composes.similarity.lin import LinSimilarity
from composes.utils.matrix_utils import to_compatible_matrix_types


class SimilarityTest(unittest.TestCase):

    def setUp(self):
        self.v1_arr = np.array([1.0, 2.0, 3.0])
        self.v1_mat = np.mat([[1.0, 2.0, 3.0]])
        self.v1_csrmat = csr_matrix(self.v1_arr)
        self.v1_cscmat = csc_matrix(self.v1_arr)
        self.v1_dmat = DenseMatrix(self.v1_mat)
        self.v1_smat = SparseMatrix(self.v1_mat)

        self.v2 = np.array([0, -2.0, 3.0])
        self.v3 = np.array([0.0, 0.0, 0.0])

        self.m1 = np.array([1.0, 2.0, 3.0])
        self.m2 = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    def test_to_compatible_matrix_type(self):
        test_cases = [(self.v1_arr, self.v1_mat, DenseMatrix),
                      (self.v1_cscmat, self.v1_mat, SparseMatrix),
                      (self.v1_mat, self.v1_cscmat, DenseMatrix),
                      (self.v1_dmat, self.v1_cscmat, DenseMatrix),
                      (self.v1_mat, self.v1_smat, SparseMatrix),
                      (self.v1_dmat, self.v1_smat, DenseMatrix),
                      (self.v1_smat, self.v1_dmat, SparseMatrix)
                      ]
        expected_output = 14.0
        s = DotProdSimilarity()
        for v1, v2, expected_matrix_type in test_cases:
            out = s.get_sim(v1, v2)
            self.assertEqual(expected_output, out)
            v1, v2 = to_compatible_matrix_types(v1, v2)
            self.assertIsInstance(v1, expected_matrix_type)
            self.assertIsInstance(v2, expected_matrix_type)

    def test_get_sim(self):
        test_cases = [(DotProdSimilarity(), self.v1_arr, self.v1_arr, 14.0),
                      (DotProdSimilarity(), self.v1_arr, self.v2, 5.0),
                      (DotProdSimilarity(), self.v1_arr, self.v3, 0.0),
                      (CosSimilarity(), self.v1_arr, self.v1_arr, 1.0),
                      (CosSimilarity(), self.v1_arr, self.v2, 0.3706246583305),
                      (CosSimilarity(), self.v1_arr, self.v3, 0.0),
                      (LinSimilarity(), self.v1_arr, self.v1_arr, 1.0),
                      (LinSimilarity(), self.v1_arr, self.v2, 0.85714285714285),
                      (LinSimilarity(), self.v1_arr, self.v3, 0.0),
                      (EuclideanSimilarity(), self.v1_arr, self.v1_arr, 1.0),
                      (EuclideanSimilarity(), self.v1_arr, self.v2, 0.19519410),
                      (EuclideanSimilarity(), self.v1_arr, self.v3, 0.21089672)
                      ]
        for sim, v1, v2, expected_output in test_cases:
            #print sim.name
            t1 = v1.copy()
            t2 = v2.copy()
            out = sim.get_sim(v1, v2)
            self.assertAlmostEqual(out, expected_output, 7)
            np.testing.assert_array_equal(v1, t1)
            np.testing.assert_array_equal(v2, t2)

    def test_get_sims_to_matrix(self):
        test_cases = [(DotProdSimilarity(), self.m1, self.v1_arr, [[14.0]]),
                      (DotProdSimilarity(), self.m2, self.v1_arr, [[14.0], [14.0]]),
                      (DotProdSimilarity(), self.m2, self.v2, [[5.0], [5.0]]),
                      (DotProdSimilarity(), self.m2, self.v3, [[0.0], [0.0]]),
                      (CosSimilarity(), self.m1, self.v1_arr, [[1.0]]),
                      (CosSimilarity(), self.m2, self.v2, [[0.370624], [0.370624]]),
                      (CosSimilarity(), self.m2, self.v3, [[0.0], [0.0]]),
                      (LinSimilarity(), self.m1, self.v1_arr, [[1.0]]),
                      (LinSimilarity(), self.m2, self.v2, [[0.857142], [0.857142]]),
                      (LinSimilarity(), self.m2, self.v3, [[0.0], [0.0]]),
                      (EuclideanSimilarity(), self.m1, self.v1_arr, [[1.0]]),
                      (EuclideanSimilarity(), self.m2, self.v2, [[0.195194], [0.195194]]),
                      (EuclideanSimilarity(), self.m2, self.v3, [[0.210896], [0.210896]])]

        for sim, in_mat, v, expected_list in test_cases:
            #print sim.name
            tv = v.copy()
            tmat = in_mat.copy()
            out = sim.get_sims_to_matrix(v, in_mat)
            np.testing.assert_array_almost_equal(np.mat(expected_list),
                                                 out.mat)
            np.testing.assert_array_equal(tv, v)
            np.testing.assert_array_equal(tmat, in_mat)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
