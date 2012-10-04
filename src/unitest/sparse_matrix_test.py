'''
Created on Sep 18, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
import numpy.testing 
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix 
from composes.matrix.sparse_matrix import SparseMatrix
from composes.matrix.dense_matrix import DenseMatrix



class TestSparseMatrix(unittest.TestCase):
    
    def setUp(self):
        self.a = np.array([[1,2,3],[4,0,5]])
        self.b = np.array([[0,0,0],[0,0,0]])
        
        self.c = np.array([[0,0],[0,0],[0,0]])
        self.d = np.array([[1,0],[0,1]])
        self.e = np.array([1,10])
        self.f = np.array([1,10,100])
        
        self.matrix_a = SparseMatrix(self.a)
        self.matrix_b = SparseMatrix(self.b)
        
        self.matrix_c = SparseMatrix(self.c)
        self.matrix_d = SparseMatrix(self.d)
        
        
    def tearDown(self):
        pass

    def test_init(self):
        nparr = self.a
        test_cases = [nparr,
                   np.mat(nparr),
                   csr_matrix(nparr),
                   csc_matrix(nparr),
                   DenseMatrix(nparr)]
        
        for inmat in test_cases:
            outmat = SparseMatrix(inmat)
            self.assertIsInstance(outmat.mat, csr_matrix)
            numpy.testing.assert_array_equal(nparr,
                                             np.array(outmat.mat.todense()))
            
    def test_add(self):
        test_cases = [(self.matrix_a, self.matrix_a, np.mat([[2,4,6],[8,0,10]])),
                      (self.matrix_a, self.matrix_b, np.mat(self.a))
                      ]
        
        for (term1, term2, expected) in test_cases:
            sum_ = term1 + term2
            numpy.testing.assert_array_equal(sum_.mat.todense(), expected)
            self.assertIsInstance(sum_, type(term1))

    def test_add_raises(self):
        test_cases = [(self.matrix_a, self.a),
                      (self.matrix_a, DenseMatrix(self.a))]

        for (term1, term2) in test_cases:
            self.assertRaises(TypeError, term1.__add__, term2)
    
    def test_mul(self):
        test_cases = [(self.matrix_a, self.matrix_c, np.mat([[0,0],[0,0]])),
                      (self.matrix_d, self.matrix_a, self.matrix_a.mat.todense()),
                      (self.matrix_a, 2, np.mat([[2,4,6],[8,0,10]])),
                      (self.matrix_a, np.int64(2), np.mat([[2,4,6],[8,0,10]]))
                      ]
        
        for (term1, term2, expected) in test_cases:
            sum_ = term1 * term2
            numpy.testing.assert_array_equal(sum_.mat.todense(), expected)
            self.assertIsInstance(sum_, type(term1))
    
    def test_mul_raises(self):
        test_cases = [(self.matrix_a, self.a),
                      (self.matrix_a, DenseMatrix(self.a)),
                      (self.matrix_a, "3")]

        for (term1, term2) in test_cases:
            self.assertRaises(TypeError, term1.__mul__, term2)

    def test_get_item(self):
        
        out_mat = SparseMatrix(self.a)[0,:]
        np.testing.assert_array_equal(out_mat.mat.todense(),np.mat(self.a[0,:]))
        
        out_int = SparseMatrix(self.a)[0,1]
        self.assertEqual(out_int, 2)
        
        out_mat = SparseMatrix(self.a)[0,1:2]
        np.testing.assert_array_equal(out_mat.mat.todense(),np.mat(self.a[0,1:2]))
        
        out_mat = SparseMatrix(self.a)[0]
        np.testing.assert_array_equal(out_mat.mat.todense(),np.mat(self.a[0,:]))
        
        
    def test_scale_rows(self):
        outcome = np.mat([[1,2,3],[40,0,50]])
        test_cases = [(self.matrix_a.copy(), self.e, outcome),
                      (self.matrix_a.copy(), np.mat(self.e).T, outcome),
                      ]
        
        for (term1, term2, expected) in test_cases:
            term1 = term1.scale_rows(term2)
            numpy.testing.assert_array_equal(term1.mat.todense(), expected)
        
    def test_scale_columns(self):
        test_cases = [(self.matrix_a.copy(), self.f, np.mat([[1,20,300],[4,0,500]]))]
        
        for (term1, term2, expected) in test_cases:
            term1 = term1.scale_columns(term2)
            numpy.testing.assert_array_equal(term1.mat.todense(), expected)
            self.assertIsInstance(term1.mat, csr_matrix)
            
    def test_scale_raises(self):
        test_cases = [(self.matrix_a, self.f, ValueError, self.matrix_a.scale_rows),
                      (self.matrix_a, self.e, ValueError, self.matrix_a.scale_columns),
                      (self.matrix_a, self.b, ValueError, self.matrix_a.scale_rows),
                      (self.matrix_a, self.b, ValueError, self.matrix_a.scale_columns),
                      (self.matrix_a, "3", TypeError, self.matrix_a.scale_rows),
                      ]                      
        for (term1, term2, error_type, function) in test_cases:
            self.assertRaises(error_type, function, term2)
            
    def test_plog(self):
        m = SparseMatrix(np.mat([[0.5,1.0,1.5],[2.0,0.0,2.5]]))
        m_expected = np.mat([[0.,0.,0.4054],[ 0.6931,0.,0.9162]])
        a_expected = np.mat([[0.,0.6931,1.0986],[1.3862,0.,1.6094]])
        test_cases = [(self.matrix_a.copy(), a_expected),
                      (m, m_expected)
                     ]

        for (term, expected) in test_cases:
            term.plog()
            numpy.testing.assert_array_almost_equal(term.mat.todense(), expected, 3)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()