'''
Created on Sep 20, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
import numpy.testing
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.transformation.scaling.epmi_weighting import EpmiWeighting
from composes.transformation.scaling.plog_weighting import PlogWeighting
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plmi_weighting import PlmiWeighting
from composes.transformation.scaling.row_normalization import RowNormalization
from composes.transformation.scaling.normalization import Normalization

class Test(unittest.TestCase):


    def setUp(self):
        self.a = np.array([[1,2,3],[4,0,5]])
        self.b = np.array([[1,2,3]])
        
        self.c = np.array([[0,0],[0,0],[0,0]])
        self.d = np.array([[1,-1],[0,1]])
        
        self.e = np.array([[1,2,3],[1,0,0]])
        self.f = np.array([1,10,100])
        
        

    def tearDown(self):
        pass


    def single_case_test(self, matrix_, expected, w):

        matrix_copy = matrix_.copy()
        dm = DenseMatrix(matrix_)
        sm = SparseMatrix(matrix_)
    
        out1 = w.apply(dm)
        out2 = w.apply(sm)
        
        numpy.testing.assert_array_almost_equal(out1.mat, expected, 7)
        numpy.testing.assert_array_almost_equal(out2.mat.todense(), expected, 7)
    
        numpy.testing.assert_array_equal(dm.mat, matrix_copy)
        numpy.testing.assert_array_equal(matrix_, matrix_copy)
        numpy.testing.assert_array_equal(sm.mat.todense(), matrix_copy)
        
    def single_case_raises_test(self, matrix_, error_type, w):
        dm = DenseMatrix(matrix_)
        sm = SparseMatrix(matrix_)
        
        self.assertRaises(error_type, w.apply, dm)
        self.assertRaises(error_type, w.apply, sm)

            
    def test_epmi(self):
        w = EpmiWeighting()
        test_cases = [(self.b, np.mat([[1,1,1]])),
                      (self.c, self.c)
                      ]
        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w)
        
    def test_plog(self):
        w = PlogWeighting()
        test_cases = [(np.mat([[1,1,1]]), np.mat([[0,0,0]])),
                      (self.c, self.c)
                      ]
        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w)
                    
    def test_ppmi(self):
        w = PpmiWeighting()
        test_cases = [(self.b, np.mat([[0,0,0]])),
                      (self.c, self.c)
                      ]
        
        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w)
            
            
    def test_plmi(self):
        w = PlmiWeighting()
        test_cases = [(self.b, np.mat([[0,0,0]])),
                      (self.c, self.c),
                      (self.e, np.mat([[0.,0.30830136,0.46245204],
                                              [1.25276297,0.,0.]]))
                      ]

        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w)
    
    def test_row_norm(self):
        w = RowNormalization()
        test_cases = [(self.b, np.mat([[0.26726124,0.53452248,0.80178373]])),
                      (self.c, self.c),
                      (self.e, np.mat([[0.26726124,0.53452248,0.80178373],
                                       [1.,0.,0.]]))
                      ]

        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w)
        
        w = RowNormalization(criterion = "length")
        test_cases = [(self.b, np.mat([[0.26726124,0.53452248,0.80178373]])),
                      (self.c, self.c),
                      (self.e, np.mat([[0.26726124,0.53452248,0.80178373],
                                       [1.,0.,0.]]))
                      ]
        
        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w) 
            
        w = RowNormalization(criterion = "sum")
        test_cases = [(self.b, np.mat([[0.16666667,0.33333333,0.5]])),
                      (self.c, self.c),
                      (self.e, np.mat([[0.16666667,0.33333333,0.5],
                                       [1.,0.,0.]]))
                      ]

        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w)
            
    def test_norm(self):
        w = Normalization()
        test_cases = [(self.b, np.mat([[1/6.0,2/6.0,3/6.0]])),
                      (self.c, self.c),
                      (self.e, np.mat([[1/7.0,2/7.0,3/7.0],
                                       [1./7.0,0.,0.]]))
                      ]

        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w)
        
        w = Normalization(criterion = "length")
        test_cases = [(self.b, np.mat([[0.26726124,0.53452248,0.80178373]])),
                      (self.c, self.c),
                      (self.e, np.mat([[0.25819889,0.51639778,0.77459667],
                                       [0.25819889,0.        ,0.        ]]))
                      ]
        
        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w) 
           
            
        w = Normalization(criterion = "sum")
        test_cases = [(self.b, np.mat([[1/6.0,2/6.0,3/6.0]])),
                      (self.c, self.c),
                      (self.e, np.mat([[1/7.0,2/7.0,3/7.0],
                                       [1./7.0,0.,0.]]))
                      ]

        for matrix_, expected in test_cases:
            self.single_case_test(matrix_, expected, w)                       
    
    def test_epmi_raises(self):    
        w = EpmiWeighting()
        test_cases = [(self.d, ValueError)]
            
        for matrix_, error_type in test_cases:
            self.single_case_raises_test(matrix_, error_type, w)
            
    def test_ppmi_raises(self):    
        w = PpmiWeighting()
        test_cases = [(self.d, ValueError)]
            
        for matrix_, error_type in test_cases:
            self.single_case_raises_test(matrix_, error_type, w)
    
    def test_plmi_raises(self):    
        w = PlmiWeighting()
        test_cases = [(self.d, ValueError)]
            
        for matrix_, error_type in test_cases:
            self.single_case_raises_test(matrix_, error_type, w)

    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_epmi']
    unittest.main()