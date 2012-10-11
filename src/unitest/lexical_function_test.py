'''
Created on Oct 11, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.semantic_space.space import Space
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.composition.lexical_function import LexicalFunction

class LexicalFunctionTest(unittest.TestCase):


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
        

        self.n_space = Space(DenseMatrix(np.mat([[3,4],[5,6]])),
                              ["car", "man"], self.ft)
        self.an_space = Space(DenseMatrix(np.mat([[3,4],[5,6]])),
                              ["a1_car", "a1_man"], self.ft)

    def tearDown(self):
        pass


    def test_simple_train(self):
        
        #TODO test a1_car twice in the phrase list
        train_data = [("a1", "car", "a1_car"),
                      ("a1", "man", "a1_man"),
                      ]
        #model with train and then compose
        model = LexicalFunction(learner=LstsqRegressionLearner(intercept=False))
  
        new_space = model.learn_function_space(train_data, self.n_space,
                                               self.an_space)
        
        np.testing.assert_array_almost_equal(new_space.cooccurrence_matrix.mat, 
                                             np.mat([1,0,0,1]), 10)
        self.assertTupleEqual(new_space.element_shape, (2,2))
        self.assertListEqual(new_space.id2row, ["a1"])
        self.assertListEqual(new_space.id2column, [])
        
        comp_space = model.compose(train_data, (new_space, self.n_space))
        
        np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat,
                                self.an_space.cooccurrence_matrix.mat, 10
                                )
        
        self.assertListEqual(comp_space.id2row, ["a1_car", "a1_man"])
        self.assertListEqual(comp_space.id2column, self.ft)
        
        #new model, without training 
        model2 = LexicalFunction()
        comp_space = model2.compose(train_data, (new_space, self.n_space))
        
        self.assertListEqual(comp_space.id2row, ["a1_car", "a1_man"])
        self.assertListEqual(comp_space.id2column, [])
        np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat,
                                             self.n_space.cooccurrence_matrix.mat,
                                             8)
        
        #recursive application
        comp_space2 = model2.compose([("a1", "a1_car", "a1_a1_car"),
                                      ("a1", "a1_man", "a1_a1_man")], 
                                     (new_space, comp_space))
        
        self.assertListEqual(comp_space2.id2row, ["a1_a1_car", "a1_a1_man"])
        self.assertListEqual(comp_space.id2column, [])
        
        np.testing.assert_array_almost_equal(comp_space2.cooccurrence_matrix.mat,
                                             self.n_space.cooccurrence_matrix.mat,
                                             8)
        self.assertEqual(comp_space.element_shape, (2,))
        self.assertEqual(comp_space2.element_shape, (2,))
      
    def test_train(self):
        
        a1_mat = DenseMatrix(np.mat([[3,4],[5,6]]))
        a2_mat = DenseMatrix(np.mat([[1,2],[3,4]]))
                             
        train_data = [("a1", "man", "a1_man"),
                      ("a2", "car", "a2_car"),
                      ("a1", "boy", "a1_boy"),
                      ("a2", "boy", "a2_boy")
                      ]        
        
        n_mat = DenseMatrix(np.mat([[13,21],[3,4],[5,6]]))
        n_space = Space(n_mat, ["man", "car", "boy"], self.ft)
        
        an1_mat = (a1_mat * n_mat.transpose()).transpose()
        an2_mat = (a2_mat * n_mat.transpose()).transpose()
        an_mat = an1_mat.vstack(an2_mat)
        
        an_space = Space(an_mat, ["a1_man","a1_car","a1_boy","a2_man","a2_car","a2_boy"], self.ft)

        #test train
        model = LexicalFunction(learner=LstsqRegressionLearner(intercept=False))
        a_space = model.learn_function_space(train_data, n_space, an_space)
        
        a1_mat.reshape((1,4))
        np.testing.assert_array_almost_equal(a1_mat.mat,
                                             a_space.cooccurrence_matrix.mat[0])

        a2_mat.reshape((1,4))
        np.testing.assert_array_almost_equal(a2_mat.mat,
                                             a_space.cooccurrence_matrix.mat[1])
        
        self.assertListEqual(a_space.id2row, ["a1", "a2"])
        self.assertTupleEqual(a_space.element_shape, (2,2))
        
        #test compose
        a1_mat = DenseMatrix(np.mat([[3,4,5,6]]))
        a2_mat = DenseMatrix(np.mat([[1,2,3,4]]))
        a_mat = a1_mat.vstack(a2_mat)
        
        a_space = Space(a_mat, ["a1", "a2"], [], element_shape=(2,2))
        model = LexicalFunction()
        comp_space = model.compose(train_data, (a_space, n_space))
        
        self.assertListEqual(comp_space.id2row, ["a1_man", "a2_car", "a1_boy", "a2_boy"])
        self.assertListEqual(comp_space.id2column, [])
        
        self.assertEqual(comp_space.element_shape, (2,))
        
        np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat,
                                             an_mat[[0,4,2,5]].mat, 8)
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()