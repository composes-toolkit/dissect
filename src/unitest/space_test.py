'''
Created on Sep 26, 2012

@author: georgianadinu
'''
import numpy as np
import unittest
from composes.semantic_space.operation import WeightingOperation
from composes.weighting.epmi import EpmiWeighting
from composes.dim_reduction.svd import Svd
from composes.dim_reduction.nmf import Nmf
from composes.matrix.dense_matrix import DenseMatrix
from composes.semantic_space.space import Space


class Test(unittest.TestCase):

    def setUp(self):
        self.init_test_cases = [(DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"],
                       ["feat1", "feat2"],
                       {"man":1, "car":0},
                       {"feat1":0, "feat2":1},
                       [WeightingOperation(EpmiWeighting())]),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"],
                       [],
                       {"man":1, "car":0},
                       {},
                       [WeightingOperation(EpmiWeighting())])]
        
        self.m1 = np.array([[1,2,3]])
        self.row1 = ["a"]
        self.row2 = ["a", "b", "c"]
        self.ft1 = ["f1","f2","f3"]
        self.space1 = Space(DenseMatrix(self.m1),self.row1, self.ft1)
    
        self.x = np.mat([[1,2,3],[2,4,6],[4,675,43]])
        self.us = np.mat([[  2.19272110e+00,   3.03174768e+00],
                               [  4.38544220e+00,   6.06349536e+00],
                               [  6.76369708e+02,  -4.91431927e-02]])
        self.space2 = Space(DenseMatrix(self.x), self.row2, self.ft1)

        
    def test_init1(self):
        for (m, id2row, id2col, row2id, col2id, ops) in self.init_test_cases:
            space_ = Space(m, id2row, id2col)
            self.assertIs(m, space_.cooccurrence_matrix)
            self.assertIs(id2row, space_.id2row)
            self.assertIs(id2col, space_.id2column)
            self.assertDictEqual(row2id, space_.row2id)
            self.assertDictEqual(col2id, space_.column2id)
            self.assertListEqual([], space_.operations)
    
    def test_init2(self):
        for (m, id2row, id2col, row2id, col2id, ops) in self.init_test_cases:
            space_ = Space(m, id2row, id2col, row2id, col2id)
            self.assertIs(m, space_.cooccurrence_matrix)
            self.assertIs(id2row, space_.id2row)
            self.assertIs(id2col, space_.id2column)
            self.assertIs(row2id, space_.row2id)
            self.assertIs(col2id, space_.column2id)
            self.assertListEqual([], space_.operations)
        
    def test_init3(self):    
        for (m, id2row, id2col, row2id, col2id, ops) in self.init_test_cases:
            space_ = Space(m, id2row, id2col, operations = ops)
            self.assertIs(m, space_.cooccurrence_matrix)
            self.assertIs(id2row, space_.id2row)
            self.assertIs(id2col, space_.id2column)
            self.assertDictEqual(row2id, space_.row2id)
            self.assertDictEqual(col2id, space_.column2id)
            self.assertListEqual(ops, space_.operations)
            
    def test_init4(self):        
        for (m, id2row, id2col, row2id, col2id, ops) in self.init_test_cases:
            space_ = Space(m, id2row, id2col, row2id, col2id, operations = ops)
            self.assertIs(m, space_.cooccurrence_matrix)
            self.assertIs(id2row, space_.id2row)
            self.assertIs(id2col, space_.id2column)
            self.assertIs(row2id, space_.row2id)
            self.assertIs(col2id, space_.column2id)
            self.assertIs(ops, space_.operations)

    def test_init_raise(self):
        test_cases = [(DenseMatrix(np.array([[1,2],[3,4],[5,6]])),
                       ["car", "man"], ["feat1", "feat2"],
                       {"man":1, "car":0}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       [], ["feat1", "feat2"],
                       {"man":1, "car":0}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1", "feat2"],
                       {}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1"],
                       {"man":1, "car":0}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1"],
                       {"man":1, "car":0}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1","feat2"],
                       {"man":1, "car":0}, {"feat1":0}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1","feat2"],
                       {"man":1, "car":0}, {"feat1":1,"feat2":0})
                      ]
        
        for (m, id2row, id2col, row2id, col2id) in test_cases:
            self.assertRaises(ValueError, Space, m, id2row, id2col,
                              row2id, col2id)
    
    def test_apply_weighting_operation(self):
        test_cases = [(self.space1, np.array([[1,1,1]]))]
        w = EpmiWeighting()
        for in_s, expected_mat in test_cases:
            out_s = in_s.apply(w)
            np.testing.assert_array_almost_equal(expected_mat, 
                                                 out_s.cooccurrence_matrix.mat,
                                                 7)
            self.assertListEqual(out_s.id2row, in_s.id2row)
            self.assertListEqual(out_s.id2column, in_s.id2column)
            self.assertDictEqual(out_s.row2id, in_s.row2id)
            self.assertDictEqual(out_s.column2id, in_s.column2id)
            self.assertEqual(1, len(out_s.operations))
            
    def test_apply_svd_reduction(self):
        test_cases = [(self.space2, self.us)]
        red = Svd(2)
        for in_s, expected_mat in test_cases:
            out_s = in_s.apply(red)
            np.testing.assert_array_almost_equal(expected_mat, 
                                                 out_s.cooccurrence_matrix.mat,
                                                 2)
            self.assertListEqual(out_s.id2row, in_s.id2row)
            self.assertListEqual(out_s.id2column, [])
            self.assertDictEqual(out_s.row2id, in_s.row2id)
            self.assertDictEqual(out_s.column2id, {})
            self.assertEqual(1, len(out_s.operations))
          
    def test_apply_nmf_reduction(self):
        test_cases = [(self.space2, self.us)]
        red = Nmf(2)
        
        for in_s, expected_mat in test_cases:
            out_s = in_s.apply(red)
            self.assertListEqual(out_s.id2row, in_s.id2row)
            self.assertListEqual(out_s.id2column, [])
            self.assertDictEqual(out_s.row2id, in_s.row2id)
            self.assertDictEqual(out_s.column2id, {})
            self.assertEqual(1, len(out_s.operations))
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()