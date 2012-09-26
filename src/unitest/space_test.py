'''
Created on Sep 26, 2012

@author: georgianadinu
'''
import numpy as np
import unittest
from composes.operation import WeightingOperation
from composes.weighting.epmi import EpmiWeighting
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
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()