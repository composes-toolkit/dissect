'''
Created on Oct 5, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.feature_selection.top_feature_selection import TopFeatureSelection
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.semantic_space.space import Space
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.weighting.plog import PlogWeighting


class Test(unittest.TestCase):

    def setUp(self):
        self.a = np.array([[1,2,3],[4,0,5]]) 
        self.space_s = Space(SparseMatrix(np.mat(self.a)), 
                                         ["a", "b"], ["f1","f2", "f3"])    

        self.space_d = Space(DenseMatrix(np.mat(self.a)), 
                                         ["a", "b"], ["f1","f2", "f3"])    

    def test_top_feat_selection(self):
        test_cases = [(self.a, np.mat([[3,1],[5,4]]), [2,0], 2), 
                      (self.a, np.mat([[3],[5]]), [2], 1),
                      (self.a, np.mat([[3,1,2],[5,4,0]]), [2,0,1], 6),
                      ]
        
        for in_mat, expected_mat, expected_perm, no_cols in test_cases:
            fs = TopFeatureSelection(no_cols)
            
            out_mat, perm = fs.apply(DenseMatrix(in_mat))
            np.testing.assert_array_equal(out_mat.mat, expected_mat)
            self.assertListEqual(perm, expected_perm)
            
            out_mat, perm = fs.apply(SparseMatrix(in_mat))
            np.testing.assert_array_equal(out_mat.mat.todense(), expected_mat)
            self.assertListEqual(perm, expected_perm)
            
            fs = TopFeatureSelection(no_cols, criterion="length")
            
            out_mat, perm = fs.apply(DenseMatrix(in_mat))
            np.testing.assert_array_equal(out_mat.mat, expected_mat)
            self.assertListEqual(perm, expected_perm)
            
            out_mat, perm = fs.apply(SparseMatrix(in_mat))
            np.testing.assert_array_equal(out_mat.mat.todense(), expected_mat)
            self.assertListEqual(perm, expected_perm)
      
        self.assertRaises(ValueError, TopFeatureSelection, 0)
        self.assertRaises(ValueError, TopFeatureSelection, 2, criterion="something")

    def test_space_top_feat_selection(self):
        
        test_cases = [(self.space_s, self.space_d, 1, ["f3"], {"f3":0}, 
                       np.mat([[3],[5]])),
                      (self.space_s, self.space_d, 2, ["f3", "f1"], {"f3":0, "f1":1}, 
                       np.mat([[3,1],[5,4]])),
                      (self.space_s, self.space_d, 4, ["f3", "f1", "f2"], 
                       {"f3":0, "f1":1, "f2":2}, 
                       np.mat([[3,1,2],[5,4,0]]))
                      ]
        
        for space_s, space_d, no_dim, id2col, col2id, mat in test_cases:
            
            trans = TopFeatureSelection(no_dim)
            new_space = space_s.apply(trans)
            
            self.assertListEqual(space_s.id2row, new_space.id2row)
            self.assertDictEqual(space_s.row2id, new_space.row2id)
            
            self.assertListEqual(id2col, new_space.id2column)
            self.assertDictEqual(col2id, new_space.column2id)
            
            np.testing.assert_array_equal(new_space.cooccurrence_matrix.mat.todense(),
                                           mat)
            self.assertEqual(len(new_space.operations), len(space_s.operations) + 1)
  
            new_space = space_d.apply(trans)
            
            self.assertListEqual(space_s.id2row, new_space.id2row)
            self.assertDictEqual(space_s.row2id, new_space.row2id)
            
            self.assertListEqual(id2col, new_space.id2column)
            self.assertDictEqual(col2id, new_space.column2id)
            
            np.testing.assert_array_equal(new_space.cooccurrence_matrix.mat,
                                           mat)
            self.assertEqual(len(new_space.operations), len(space_s.operations) + 1)          
        
        
        self.assertRaises(ValueError, TopFeatureSelection, 0)
        
    def test_per_space_top_feat_selection(self):
                
        test_cases = [(self.space_d, 1, ["f3"], {"f3":0}, 
                       np.mat([[3],[5]])),
                      (self.space_d, 2, ["f3", "f1"], {"f3":0, "f1":1}, 
                       np.mat([[3,1],[5,4]])),
                      (self.space_d, 4, ["f3", "f1", "f2"], 
                       {"f3":0, "f1":1, "f2":2}, 
                       np.mat([[3,1,2],[5,4,0]]))
                      ]

        for space_d, no_dim, id2col, col2id, mat in test_cases:
            
            trans = TopFeatureSelection(no_dim)
            new_space = space_d.apply(trans)
            
            #peripheral test simple test
            per_sp = PeripheralSpace(new_space, DenseMatrix(self.a), ["c","d"])

            self.assertListEqual(per_sp.id2row, ["c","d"])
            self.assertListEqual(per_sp.id2column, id2col)           
            self.assertDictEqual(per_sp.column2id, col2id)
            
            np.testing.assert_array_equal(per_sp.cooccurrence_matrix.mat,
                                          mat)
            
            #peripheral test with add rows
            per_sp = PeripheralSpace(new_space, DenseMatrix(self.a[0,:]), ["c"])
            per_sp.add_rows(DenseMatrix(self.a[1,:]), ["d"])
            
            self.assertListEqual(per_sp.id2row, ["c","d"])
            self.assertListEqual(per_sp.id2column, id2col)           
            self.assertDictEqual(per_sp.column2id, col2id)
            
            np.testing.assert_array_equal(per_sp.cooccurrence_matrix.mat,
                                          mat)
            
            #peripheral test, with plog applied to core BEFORE feat selection
            plogmat = mat.copy() 
            plogmat[plogmat==0] = 1
            plogmat = np.log(plogmat)
            
            new_space = space_d.apply(PlogWeighting())
            trans = TopFeatureSelection(no_dim)
            new_space = new_space.apply(trans)

            per_sp = PeripheralSpace(new_space, DenseMatrix(self.a), ["c","d"])

            self.assertListEqual(per_sp.id2row, ["c","d"])
            self.assertListEqual(per_sp.id2column, id2col)           
            self.assertDictEqual(per_sp.column2id, col2id)
            
            np.testing.assert_array_almost_equal(per_sp.cooccurrence_matrix.mat,
                                                 plogmat, 7)
        
            #peripheral test, with plog applied to core AFTER feat selection 
            trans = TopFeatureSelection(no_dim)
            new_space = space_d.apply(trans)
            new_space = new_space.apply(PlogWeighting())

            per_sp = PeripheralSpace(new_space, DenseMatrix(self.a), ["c","d"])

            self.assertListEqual(per_sp.id2row, ["c","d"])
            self.assertListEqual(per_sp.id2column, id2col)           
            self.assertDictEqual(per_sp.column2id, col2id)
            
            np.testing.assert_array_almost_equal(per_sp.cooccurrence_matrix.mat,
                                                 plogmat, 7)
                            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_top_feat_selection']
    unittest.main()