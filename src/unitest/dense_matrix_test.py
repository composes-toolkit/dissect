'''
Created on Sep 17, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
import numpy.testing 
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix 
from composes.matrix.sparse_matrix import SparseMatrix
from composes.matrix.dense_matrix import DenseMatrix
    
class TestDenseMatrix(unittest.TestCase):
    
    
    def setUp(self):
        self.a = np.array([[1,2,3],[4,0,5]])
        self.b = np.array([[0,0,0],[0,0,0]])
        
        self.c = np.array([[0,0],[0,0],[0,0]])
        self.d = np.array([[1,0],[0,1]])
        self.e = np.array([1,10])
        self.f = np.array([1,10,100])
        
        self.matrix_a = DenseMatrix(self.a)
        self.matrix_b = DenseMatrix(self.b)
        
        self.matrix_c = DenseMatrix(self.c)
        self.matrix_d = DenseMatrix(self.d)

    def tearDown(self):
        pass

    def test_init(self):
        nparr = self.a
        test_cases = [nparr,
                   np.mat(nparr),
                   csr_matrix(nparr),
                   csc_matrix(nparr),
                   SparseMatrix(nparr)]
        
        for inmat in test_cases:
            outmat = DenseMatrix(inmat)
            self.assertIsInstance(outmat.mat, np.matrix)
            numpy.testing.assert_array_equal(nparr, np.array(outmat.mat))


    def test_add(self):
        test_cases = [(self.matrix_a, self.matrix_a, np.mat([[2,4,6],[8,0,10]])),
                      (self.matrix_a, self.matrix_b, self.matrix_a.mat)
                      ]

        
        for (term1, term2, expected) in test_cases:
            sum_ = term1 + term2
            numpy.testing.assert_array_equal(sum_.mat, expected)
            self.assertIsInstance(sum_, type(term1))

    def test_add_raises(self):
        test_cases = [(self.matrix_a, self.a),
                      (self.matrix_a, SparseMatrix(self.a))]

        for (term1, term2) in test_cases:
            self.assertRaises(TypeError, term1.__add__, term2)
    
    def test_div(self):
        test_cases = [(self.matrix_a, 2, np.mat([[0.5,1.0,1.5],[2.0,0.0,2.5]])),
                      (self.matrix_c, 2, np.mat(self.c))
                     ]
        
        for (term1, term2, expected) in test_cases:
            sum_ = term1 / term2
            numpy.testing.assert_array_equal(sum_.mat, expected)
            self.assertIsInstance(sum_, DenseMatrix)
    
    def test_div_raises(self):
        test_cases = [(self.matrix_a, self.a, TypeError),
                      (self.matrix_a, SparseMatrix(self.a), TypeError),
                      (self.matrix_a, "3", TypeError),
                      (self.matrix_a, 0, ZeroDivisionError)
                      ]

        for (term1, term2, error_type) in test_cases:
            self.assertRaises(error_type, term1.__div__, term2)
            
            
    def test_mul(self):
        test_cases = [(self.matrix_a, self.matrix_c, np.mat([[0,0],[0,0]])),
                      (self.matrix_d, self.matrix_a, self.matrix_a.mat),
                      (self.matrix_a, 2, np.mat([[2,4,6],[8,0,10]])),
                      (2, self.matrix_a, np.mat([[2,4,6],[8,0,10]])),
                      (self.matrix_a, np.int64(2), np.mat([[2,4,6],[8,0,10]])),
                      (np.int64(2), self.matrix_a, np.mat([[2,4,6],[8,0,10]]))
                      ]
        
        for (term1, term2, expected) in test_cases:
            sum_ = term1 * term2
            numpy.testing.assert_array_equal(sum_.mat, expected)
            self.assertIsInstance(sum_, DenseMatrix)
    
    def test_mul_raises(self):
        test_cases = [(self.matrix_a, self.a),
                      (self.matrix_a, SparseMatrix(self.a)),
                      (self.matrix_a, "3"),
                      ("3", self.matrix_a)]
        
        for (term1, term2) in test_cases:
            self.assertRaises(TypeError, term1.__mul__, term2)
            
    def test_multiply(self):
        test_cases = [(self.matrix_a, self.matrix_a, np.mat([[1,4,9],[16,0,25]])),
                      (self.matrix_a, self.matrix_b, np.mat(self.b))
                      ]
        
        for (term1, term2, expected) in test_cases:
            mult1 = term1.multiply(term2)
            mult2 = term2.multiply(term1)
            
            numpy.testing.assert_array_equal(mult1.mat, expected)
            numpy.testing.assert_array_equal(mult2.mat, expected)
            
            self.assertIsInstance(mult1, DenseMatrix)
            self.assertIsInstance(mult2, DenseMatrix)             
    
    def test_multiply_raises(self):
        
        test_cases = [(self.matrix_a, self.matrix_d, ValueError),
                      (self.matrix_a, self.a, TypeError),
                      (self.matrix_a, SparseMatrix(self.a), TypeError),
                      ]
            
        for (term1, term2, error_type) in test_cases:
            self.assertRaises(error_type, term1.multiply, term2)

    def test_scale_rows(self):
        outcome = np.mat([[1,2,3],[40,0,50]])
        test_cases = [(self.matrix_a, self.e, outcome),
                      (self.matrix_a, np.mat(self.e).T, outcome),
                      ]
        
        for (term1, term2, expected) in test_cases:
            term1 = term1.scale_rows(term2)
            numpy.testing.assert_array_equal(term1.mat, expected)
        
    def test_scale_columns(self):
        test_cases = [(self.matrix_a, self.f, np.mat([[1,20,300],[4,0,500]]))]
        
        for (term1, term2, expected) in test_cases:
            term1 = term1.scale_columns(term2)
            numpy.testing.assert_array_equal(term1.mat, expected)
            
            
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
        m = DenseMatrix(np.mat([[0.5,1.0,1.5],[2.0,0.0,2.5]]))
        m_expected = np.mat([[0.,0.,0.4054],[ 0.6931,0.,0.9162]])
        a_expected = np.mat([[0.,0.6931,1.0986],[1.3862,0.,1.6094]])
        test_cases = [(self.matrix_a.copy(), a_expected),
                      (m, m_expected)
                     ]

        for (term, expected) in test_cases:
            term.plog()
            numpy.testing.assert_array_almost_equal(term.mat, expected, 3)
            
    def test_svd(self):
        test_cases = [(np.mat([[1,2,3],[2,4,6],[4,675,43]]), 
                       np.mat([[-3.2418e-03,-4.4720e-01],
                               [-6.4836e-03,-8.9440e-01],
                               [-9.9997e-01,7.2489e-03]]),
                       np.array([6.7639e+02,6.7794e+00]),
                       np.mat([[-0.0059,-0.9979,-0.0636],
                               [-0.3255,0.0621,-0.9434]]).transpose())
                        ]
        for x, u_expected, s_expected, v_expected in test_cases:
            u, s, v = DenseMatrix(x).svd(2)
            np.testing.assert_array_almost_equal(u.mat, u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat, v_expected, 2)

            u, s, v = DenseMatrix(x).svd(3)
            np.testing.assert_array_almost_equal(u.mat, u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat, v_expected, 2)
            
            u, s, v = DenseMatrix(x).svd(6)
            np.testing.assert_array_almost_equal(u.mat, u_expected, 2)
            np.testing.assert_array_almost_equal(s, s_expected, 2)
            np.testing.assert_array_almost_equal(v.mat, v_expected, 2)
            
            u, s, v = DenseMatrix(x).svd(1)
            np.testing.assert_array_almost_equal(u.mat, u_expected[:,0:1], 2)
            np.testing.assert_array_almost_equal(s, s_expected[0:1], 2)
            np.testing.assert_array_almost_equal(v.mat, v_expected[:, 0:1], 2)


    def test_svd_raises(self):
        test_cases = [np.mat([[1,2,3],[2,4,6],[4,675,43]])]
        
        for x in test_cases:   
            self.assertRaises(ValueError, DenseMatrix(x).svd, 0)
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()