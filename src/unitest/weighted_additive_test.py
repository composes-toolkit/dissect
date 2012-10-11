'''
Created on Oct 5, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.composition.weighted_additive import WeightedAdditive
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.semantic_space.space import Space

class WeightedAdditiveTest(unittest.TestCase):


    def setUp(self):
        self.m11 = DenseMatrix(np.mat([[3],[9]]))
        self.m21 = DenseMatrix(np.mat([[4],[2]]))
        self.ph1 = DenseMatrix(np.mat([[18],[24]]))

        self.m12 = DenseMatrix(np.mat([[3,1],[9,2]]))
        self.m22 = DenseMatrix(np.mat([[4,3],[2,1]]))
        self.ph2 = DenseMatrix(np.mat([[18,11],[24,7]]))

        
        self.row = ["a", "b"]
        self.ft = ["f1","f2"]
        self.space1 = Space(DenseMatrix(self.m12), self.row, self.ft)
        self.space2 = Space(DenseMatrix(self.m12), self.row, [])


    def tearDown(self):
        pass


    def test_train(self):
        test_cases = [(self.m11, self.m21, self.ph1, 2, 3),
                      (self.m12, self.m22, self.ph2, 2, 3),
                      (self.m11, self.m21, DenseMatrix(np.mat([[0],[0]])), 0, 0),
                      (SparseMatrix(self.m12), SparseMatrix(self.m22),
                        SparseMatrix(self.ph2), 2, 3),
                      (self.m11, DenseMatrix(np.mat([[0],[0]])), self.ph1, 3, 0),
                      (DenseMatrix(np.mat([[0],[0]])), self.m11, self.ph1, 0, 3),
                      (DenseMatrix(np.mat([[1,2,3]])),
                       DenseMatrix(np.mat([[2,4,6]])),
                       DenseMatrix(np.mat([[3,6,9]])), 0.6, 1.2),
                      (DenseMatrix(np.mat([[0],[0]])), 
                       DenseMatrix(np.mat([[0],[0]])), 
                       DenseMatrix(np.mat([[0],[0]])), 0.0, 0.0)
                      ]
        for m1, m2, ph, expected_alpha, expected_beta in test_cases:
            model = WeightedAdditive()
            model._train(m1, m2, ph)
            self.assertAlmostEqual(model.alpha, expected_alpha, 8) 
            self.assertAlmostEqual(model.beta, expected_beta, 8) 

    def test_compose(self):
        
        model = WeightedAdditive(2,3)
        np.testing.assert_array_equal(model._compose(self.m11, self.m21).mat,
                                                      self.ph1.mat)
        
        model = WeightedAdditive()
        np.testing.assert_array_equal(model._compose(self.m11, self.m21).mat,
                                                      np.mat([[7/2.],[11/2.]]))
        
        model = WeightedAdditive(0.5)
        np.testing.assert_array_equal(model._compose(self.m11, self.m21).mat,
                                                      np.mat([[7/2.],[11/2.]]))
        
    def test_space_train(self):
        test_cases = [ ([("a", "b", "a_b"), ("a", "a", "a_a")],
                        self.space1,
                        Space(DenseMatrix(np.mat([[12,3],[6,2]])), 
                              ["a_b", "a_a"],["f1", "f2"]),
                        1, 1
                       ),
                      ([("a", "b", "a_b"), ("a", "a", "a_a")],
                        self.space1,
                        Space(DenseMatrix(np.mat([[0,0],[0,0]])), 
                              ["a_b", "a_a"],["f1", "f2"]),
                        0, 0
                       ),
                      ([("a", "b", "a_b"), ("a", "a", "a_a")],
                        self.space1,
                        Space(DenseMatrix(np.mat([[0,0],[0,0]])), 
                              ["a_b", "a_a"],[]),
                        0, 0
                       ),
                      ([("a", "b", "a_b")],
                        self.space1,
                        Space(DenseMatrix(np.mat([[21,5]])), 
                              ["a_b"],[]),
                        1, 2
                       )
                      ]    
        
        for in_data, arg_space, phrase_space, alpha, beta in test_cases: 
            model = WeightedAdditive()
            model.train(in_data, arg_space, phrase_space)
            
            self.assertAlmostEqual(model.alpha, alpha, 7)
            self.assertAlmostEqual(model.beta, beta, 7)
            
            comp_space = model.compose(in_data, arg_space)
            self.assertListEqual(comp_space.id2row, phrase_space.id2row)
            self.assertListEqual(comp_space.id2column, phrase_space.id2column)
            
            self.assertDictEqual(comp_space.row2id, phrase_space.row2id)
            self.assertDictEqual(comp_space.column2id, phrase_space.column2id)
            
            np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat,
                                                 phrase_space.cooccurrence_matrix.mat, 
                                                 8)
            
    def test_space_compose(self):
        
        test_cases = [(WeightedAdditive(1, 1), 
                       [("a", "b", "a_b"), ("a", "a", "a_a")],
                        self.space1,
                        ["a_b", "a_a"],
                        {"a_b":0, "a_a":1},
                        ["f1", "f2"],
                        {"f1":0, "f2":1},
                        np.mat([[12, 3],[6,2]])
                       ),
                      (WeightedAdditive(0, 0), 
                       [("a", "b", "a_b"), ("a", "a", "a_a")],
                        self.space1,
                        ["a_b", "a_a"],
                        {"a_b":0, "a_a":1},
                        ["f1", "f2"],
                        {"f1":0, "f2":1},
                        np.mat([[0,0],[0,0]])
                       ),
                      (WeightedAdditive(1, 1), 
                       [("a", "b", "a_b"), ("a", "a", "a_a")],
                        self.space2,
                        ["a_b", "a_a"],
                        {"a_b":0, "a_a":1},
                        [],
                        {},
                        np.mat([[12, 3],[6,2]])
                       ),
                      (WeightedAdditive(1, 2), 
                       [("a", "b", "a_b")],
                        self.space1,
                        ["a_b"],
                        {"a_b":0},
                        ["f1", "f2"],
                        {"f1":0, "f2":1},
                        np.mat([[21,5]])
                       )
                      ]

        for model, in_data, space, id2row, row2id, id2col, col2id, mat_ in test_cases:
            comp_sp = model.compose(in_data, space)
            
            self.assertListEqual(comp_sp.id2row, id2row)
            self.assertDictEqual(comp_sp.row2id, row2id)
             
            self.assertListEqual(comp_sp.id2column, id2col)
            self.assertDictEqual(comp_sp.column2id, col2id)
             
            self.assertEqual(len(comp_sp.operations), 0) 
            np.testing.assert_array_equal(comp_sp.cooccurrence_matrix.mat, mat_)
             
                                      
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()