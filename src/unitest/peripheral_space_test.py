'''
Created on Sep 27, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.semantic_space.space import Space
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.matrix.dense_matrix import DenseMatrix
from composes.transformation.scaling.epmi_weighting import EpmiWeighting
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.plog_weighting import PlogWeighting
from composes.transformation.scaling.raw_weighting import RawWeighting
from composes.transformation.dim_reduction.svd import Svd

class PeripheralSpaceTest(unittest.TestCase):

    def setUp(self):
        self.m1 = np.array([[1,2,3]])
        self.row1 = ["a"]
        self.ft1 = ["f1","f2","f3"]
        self.space1 = Space(DenseMatrix(self.m1),self.row1, self.ft1)
        
        self.m2 = np.array([[4,2,6]])
        self.row2 = ["b"]
        self.row3 = ["a", "b", "c"]
        
        self.x = np.mat([[1,2,3],[2,4,6],[4,675,43]])
        self.us = np.mat([[  2.19272110e+00,   3.03174768e+00],
                               [  4.38544220e+00,   6.06349536e+00],
                               [  6.76369708e+02,  -4.91431927e-02]])
        self.us2 =  np.mat([[ 2.19272110e+00],
                            [ 4.38544220e+00],
                            [ 6.76369708e+02]])


        self.space2 = Space(DenseMatrix(self.x), self.row3, self.ft1)
        
    def assert_column_identical(self, space1, space2):
        self.assertListEqual(space1.id2column, space2.id2column)
        self.assertDictEqual(space1.column2id, space2.column2id)
        
    def test_add_rows(self):    

        test_cases = [(self.space1, self.m2, self.row2,
                        np.array([[4,2,6]]), ["c"],
                        np.array([[4,2,6],[4,2,6]]),
                        np.array([[0.69314718,0,0],[0.69314718,0,0]]),
                        {"b":0,"c":1},
                        ["b", "c"])]
        
        for (core_sp, per_mat1, id2row1, per_mat2, id2row2, per_exp_mat1,
             per_exp_mat2, per_exp_row2id, per_exp_id2row) in test_cases:
            
            per_sp = PeripheralSpace(core_sp, DenseMatrix(per_mat1), id2row1)
            per_sp.add_rows(DenseMatrix(per_mat2), id2row2)
            np.testing.assert_array_almost_equal(per_sp.cooccurrence_matrix.mat,
                                            per_exp_mat1, 7)
            
            self.assertDictEqual(per_sp.row2id, per_exp_row2id)
            self.assertListEqual(per_sp.id2row, per_exp_id2row)
            
            self.assertDictEqual(per_sp.column2id, core_sp.column2id)
            self.assertListEqual(per_sp.id2column, core_sp.id2column)
            
            core_sp2 = core_sp.apply(PpmiWeighting())
            per_sp2 = PeripheralSpace(core_sp2, DenseMatrix(per_mat1), id2row1)
            per_sp2.add_rows(DenseMatrix(per_mat2), id2row2)
            np.testing.assert_array_almost_equal(per_sp2.cooccurrence_matrix.mat,
                                            per_exp_mat2, 7)
            
            self.assertRaises(ValueError, per_sp2.add_rows,
                              DenseMatrix(per_mat2), id2row1)
            
            self.assertRaises(ValueError, per_sp2.add_rows, 
                              DenseMatrix(per_mat2), id2row2)
        
            self.assertRaises(ValueError, per_sp2.add_rows, 
                              DenseMatrix(per_mat2), ["d", "e"])
        
        
    def test_init(self):
        test_cases = [(self.space1, self.m2, self.row2, np.array([[2,0.5,1]]), 
                       np.array([[0.69314718,0,0]]))]
        
        
        w1 = EpmiWeighting()
        w2 = PlogWeighting()
        w3 = RawWeighting()
        
        for core_s, per_mat, per_row, per_mat_out1, per_mat_out2 in test_cases:
            tmp_mat = per_mat.copy()
            tmp_core_mat = core_s.cooccurrence_matrix.mat 
            per_s1 = PeripheralSpace(core_s, DenseMatrix(per_mat), per_row)
            
            np.testing.assert_array_equal(per_s1.cooccurrence_matrix.mat, 
                                          tmp_mat)
            self.assert_column_identical(per_s1, core_s)
            self.assertListEqual(per_s1.id2row, per_row)
            self.assertListEqual(per_s1.operations, core_s.operations)
            
            core_s1 = core_s.apply(w1)
            per_s2 = PeripheralSpace(core_s1, DenseMatrix(per_mat), per_row)
            np.testing.assert_array_almost_equal(per_s2.cooccurrence_matrix.mat, 
                                          per_mat_out1)
            self.assert_column_identical(per_s2, core_s1)
            self.assertListEqual(per_s2.id2row, per_row)
            self.assertListEqual(per_s2.operations, core_s1.operations)
            self.assertEqual(len(per_s2.operations), 1)
            
            core_s2 = core_s1.apply(w2)
            per_s3 = PeripheralSpace(core_s2, DenseMatrix(per_mat), per_row)
            np.testing.assert_array_almost_equal(per_s3.cooccurrence_matrix.mat, 
                                          per_mat_out2)
            self.assert_column_identical(per_s3, core_s2)
            self.assertListEqual(per_s3.id2row, per_row)
            self.assertListEqual(per_s3.operations, core_s2.operations)
            self.assertEqual(len(per_s3.operations), 2)

            np.testing.assert_array_equal(tmp_core_mat, 
                                          core_s.cooccurrence_matrix.mat)
            
            core_s3 = core_s2.apply(w3)
            per_s4 = PeripheralSpace(core_s3, DenseMatrix(per_mat), per_row)
            np.testing.assert_array_almost_equal(per_s4.cooccurrence_matrix.mat, 
                                          per_mat_out2)
            self.assert_column_identical(per_s4, core_s2)
            self.assertListEqual(per_s4.id2row, per_row)
            self.assertListEqual(per_s4.operations, core_s3.operations)
            self.assertEqual(len(per_s4.operations), 3)

            np.testing.assert_array_equal(tmp_core_mat, 
                                          core_s.cooccurrence_matrix.mat)
            

            
    def test_init_svd(self):
        test_cases = [(self.space2, self.us, self.us2, self.x, self.row3)]
        red1 = Svd(2)
        red2 = Svd(1)
        w = RawWeighting()
        
        for in_s, expected_mat, expected_mat2, data, rows in test_cases:
            in_s = in_s.apply(red1)
            per_s = PeripheralSpace(in_s, DenseMatrix(data), rows)

            np.testing.assert_array_almost_equal(expected_mat, 
                                                 per_s.cooccurrence_matrix.mat,
                                                 2)
            self.assertListEqual(per_s.id2row, in_s.id2row)
            self.assertListEqual(per_s.id2column, [])
            self.assertDictEqual(per_s.row2id, in_s.row2id)
            self.assertDictEqual(per_s.column2id, {})
            self.assertEqual(1, len(per_s.operations))
            
            in_s = in_s.apply(red2)
            per_s = PeripheralSpace(in_s, DenseMatrix(data), rows)

            np.testing.assert_array_almost_equal(expected_mat2, 
                                                 per_s.cooccurrence_matrix.mat,
                                                 2)
            self.assertListEqual(per_s.id2row, in_s.id2row)
            self.assertListEqual(per_s.id2column, [])
            self.assertDictEqual(per_s.row2id, in_s.row2id)
            self.assertDictEqual(per_s.column2id, {})
            self.assertEqual(2, len(per_s.operations))
            
            in_s = in_s.apply(w)
            per_s = PeripheralSpace(in_s, DenseMatrix(data), rows)

            np.testing.assert_array_almost_equal(expected_mat2, 
                                                 per_s.cooccurrence_matrix.mat,
                                                 2)
            self.assertListEqual(per_s.id2row, in_s.id2row)
            self.assertListEqual(per_s.id2column, [])
            self.assertDictEqual(per_s.row2id, in_s.row2id)
            self.assertDictEqual(per_s.column2id, {})
            self.assertEqual(3, len(per_s.operations))
            
    def test_add_rows_svd(self):
        test_cases = [(self.space2, np.vstack([self.us2[0], self.us2[0]]), 
                       self.m1, ["e"], ["f"], {"e":0, "f":1})]
        red1 = Svd(2)
        red2 = Svd(1)
        
        for in_s, expected_mat, data, id2row1, id2row2, row2id in test_cases:
            in_s = in_s.apply(red1)
            in_s = in_s.apply(red2)
            per_s = PeripheralSpace(in_s, DenseMatrix(data), id2row1)
            per_s.add_rows(DenseMatrix(data), id2row2)    

            np.testing.assert_array_almost_equal(expected_mat, 
                                                 per_s.cooccurrence_matrix.mat,
                                                 2)
            self.assertListEqual(per_s.id2row, id2row1 + id2row2)
            self.assertListEqual(per_s.id2column, [])
            self.assertDictEqual(per_s.row2id, row2id)
            self.assertDictEqual(per_s.column2id, {})
            self.assertEqual(2, len(per_s.operations))
            
            
                        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()