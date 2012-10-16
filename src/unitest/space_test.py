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
from composes.matrix.sparse_matrix import SparseMatrix
from composes.semantic_space.space import Space
from composes.similarity.dot_prod import DotProdSimilarity

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
     
    def test_get_sim(self):
        test_cases = [(self.space2, None, "a", "b", 28),
                      (self.space2, None, "a", "none", 0),
                      (self.space2, self.space2, "a", "b", 28),
                      (self.space2, self.space1, "a", "a", 14),
                      (self.space2, self.space1, "a", "none", 0),
                      (self.space2, self.space1, "none", "a", 0)
                      ]
        for space, space2, w1, w2, expected_sim in test_cases:
            if space2 is None:
                sim = space.get_sim(w1, w2, DotProdSimilarity())
            else:
                sim = space.get_sim(w1, w2, DotProdSimilarity(), space2)
  
            self.assertEqual(sim, expected_sim)
 
    def test_get_sim_raises(self):
        
        space3 = self.space2.apply(Svd(2))
        self.assertRaises(ValueError, self.space2.get_sim, 
                          "a", "b", DotProdSimilarity(), space3)

    def test_get_neighbours(self):
        test_cases = [(self.space2, None, "a", 0,[]),
                      (self.space2, None, "a", 1,[("c",1483)]),
                      (self.space2, None, "a", 2,[("c",1483),("b",28)]),
                      (self.space2, None, "a", 3,[("c",1483),("b",28),("a",14)]),
                      (self.space2, None, "a", 4,[("c",1483),("b",28),("a",14)]),
                      (self.space2, self.space1, "a", 1,[("a",14)]),
                      (self.space2, self.space1, "a", 3,[("a",14)]),
                      (self.space2, self.space1, "none", 4,[])
                      ]            
        for space, space2, word, no_neigh, expected in test_cases:
            if space2 is None:
                neighbours = space.get_neighbours(word, no_neigh, 
                                                  DotProdSimilarity())
            else:
                neighbours = space.get_neighbours(word, no_neigh, 
                                                  DotProdSimilarity(),
                                                  space2)    
         
            self.assertListEqual(expected, neighbours)    
            
    def test_get_neighbours_raises(self):
        
        space3 = self.space2.apply(Svd(2))
        self.assertRaises(ValueError, self.space2.get_neighbours, 
                          "a", 2, DotProdSimilarity(), space3)
        
      
    def test_vstack(self):
        space3 = Space(DenseMatrix(self.x[0:2,:]), ["e","f"], self.ft1)
        
        space4 = Space(DenseMatrix(np.vstack((self.x, self.x[0:2,:]))),
                       ["a", "b", "c", "e","f"], self.ft1)
        
        test_cases = [(self.space2, space3, space4)]
        for space1, space2, expected_space in test_cases:
            outcome = space1.vstack(space1, space2)
            np.testing.assert_array_equal(expected_space.cooccurrence_matrix.mat,
                                          outcome.cooccurrence_matrix.mat)
            
            self.assertListEqual(outcome.id2column, space1.id2column)
            self.assertListEqual(outcome.id2column, expected_space.id2column)    

            self.assertDictEqual(outcome.column2id, space1.column2id)
            self.assertDictEqual(outcome.column2id, expected_space.column2id)
            
            self.assertListEqual(outcome.id2row, expected_space.id2row)
            self.assertDictEqual(outcome.row2id, expected_space.row2id)
            
            self.assertListEqual([], outcome.operations)
            
    def test_vstack_raises(self):
        
        space3 = Space(DenseMatrix(self.x[0:2,0:1]), ["e","f"], self.ft1[0:1])
        space4 = Space(DenseMatrix(self.x[0:2,:]), ["a","f"], self.ft1)
        space5 = Space(DenseMatrix(self.x[0:2,:]), ["e","f"], [])
        space6 = Space(DenseMatrix(self.x[0:2,:]), ["e","f"], ["f1","f2","f4"])
        
        test_cases = [(self.space2, space3),
                      (self.space2, space4),
                      (self.space2, space5),
                      (self.space2, space6)
                      ]
        
        for space1, space2 in test_cases:
            self.assertRaises(ValueError, space1.vstack, space1, space2)
            
                         
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
     
    def test_build(self):
        
        dir_ = "space_test_resources/" 
        test_cases = [("data1",["red", "blue"], ["car", "man"], 
                       np.mat([[3,5],[0,10]]))
                      ]
        for data_file, rows, cols, mat in test_cases:
            data_file1 = dir_ + data_file + ".sparse"

            sp = Space.build(data=data_file1, format="sm")
            self.assertListEqual(rows, sp.id2row)
            self.assertListEqual(cols, sp.id2column)
            
            self.assertIsInstance(sp.cooccurrence_matrix, SparseMatrix)
            np.testing.assert_array_equal(mat, 
                                          sp.cooccurrence_matrix.mat.todense())
         
            
            data_file2 = dir_ + data_file + ".dense"
                
            sp = Space.build(data=data_file2, format="dm")
            self.assertListEqual(rows, sp.id2row)
            self.assertListEqual([], sp.id2column)
                       
            self.assertIsInstance(sp.cooccurrence_matrix, DenseMatrix)
            np.testing.assert_array_equal(mat, sp.cooccurrence_matrix.mat) 
             
         
                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()