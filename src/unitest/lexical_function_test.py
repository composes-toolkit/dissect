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
        self.ft = ["f1","f2"]

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
      
    def test_simple_3d(self):
        
        train_data1 = [("drive_car", "I", "I_drive_car"),
                       ("read_man", "You", "You_read_man"),
                       ("read_man", "I", "I_read_man"),
                       ("drive_car", "You", "You_drive_car"),
                       ("drive_man", "You", "You_drive_man"),
                       ("drive_man", "I", "I_drive_man")
                       ]
        
        train_data2 = [("drive", "car", "drive_car"),
                       ("drive", "man", "drive_man"),
                       ]
        
        n_mat = DenseMatrix(np.mat([[1,2],[3,4],[5,6],[7,8]]))
        svo_mat = DenseMatrix(np.mat([[1,2],[3,4],[1,2],[3,4],[3,4],[1,2]])) 
        
        n_space = Space(n_mat,["I", "You", "man", "car"],[])
        svo_space = Space(svo_mat,["I_drive_car","You_read_man", 
                                 "I_read_man", "You_drive_car", 
                                 "You_drive_man", "I_drive_man"],["f1","f2"]) 
        
        #test first stage train
        model = LexicalFunction(learner=LstsqRegressionLearner(intercept=False))
        vo_space = model.learn_function_space(train_data1, n_space, svo_space)
        
        np.testing.assert_array_almost_equal(vo_space.cooccurrence_matrix.mat, 
                                             np.mat([[1,0,0,1],[1,0,0,1],[1,0,0,1]]), 10)
        self.assertTupleEqual(vo_space.element_shape, (2,2))
        self.assertListEqual(vo_space.id2row, ["drive_car","drive_man","read_man"])
        self.assertListEqual(vo_space.id2column, [])                        
                                
        #test first stage compose
        comp_space = model.compose([train_data1[0]], (vo_space, n_space))
        np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat, 
                                             np.mat([[1,2]]), 8)
        
        self.assertTupleEqual(comp_space.element_shape, (2,))
        self.assertListEqual(comp_space.id2row, ["I_drive_car"])
        self.assertListEqual(comp_space.id2column, ["f1","f2"])
        
        #test second stage train
        model = LexicalFunction(learner=LstsqRegressionLearner(intercept=False))  
        v_space =  model.learn_function_space(train_data2, n_space, vo_space)            
        
        np.testing.assert_array_almost_equal(v_space.cooccurrence_matrix.mat, 
                                             np.mat([[-1.,1.,0.,0.,0.,0.,-1.,1.]]), 10)
        
        self.assertTupleEqual(v_space.element_shape, (2,2,2))
        self.assertListEqual(v_space.id2row, ["drive"])
        self.assertListEqual(v_space.id2column, [])     
        
        #test compose1
        comp_space = model.compose([train_data2[0]], (v_space, n_space))
        np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat, 
                                             np.mat([[1,0,0,1]]), 8)
        
        self.assertTupleEqual(comp_space.element_shape, (2,2))
        self.assertListEqual(comp_space.id2row, ["drive_car"])
        self.assertListEqual(comp_space.id2column, [])
        
   
        #test compose2
        comp_space2 = model.compose([train_data1[0]], (comp_space, n_space))
        np.testing.assert_array_almost_equal(comp_space2.cooccurrence_matrix.mat, 
                                             np.mat([[1,2]]), 8)
        
        self.assertTupleEqual(comp_space2.element_shape, (2,))
        self.assertListEqual(comp_space2.id2row, ["I_drive_car"])
        self.assertListEqual(comp_space2.id2column, [])
        
        #recursive application, write a wrapper around it!!!
        comp_space2 = model.compose([("drive_car", "I", "I_drive_car")], (model.compose([("drive","car","drive_car")], (v_space, n_space)), n_space))
        np.testing.assert_array_almost_equal(comp_space2.cooccurrence_matrix.mat, 
                                             np.mat([[1,2]]), 8)
        
        self.assertTupleEqual(comp_space2.element_shape, (2,))
        self.assertListEqual(comp_space2.id2row, ["I_drive_car"])
        self.assertListEqual(comp_space2.id2column, [])        
        
      
    def test_3d(self):
        
        # setting up
        v_mat = DenseMatrix(np.mat([[0,0,1,1,2,2,3,3],#hate 
                                    [0,1,2,4,5,6,8,9]])) #love 
        
        
        vo11_mat = DenseMatrix(np.mat([[0,11],[22,33]])) #hate boy
        vo12_mat = DenseMatrix(np.mat([[0,7],[14,21]])) #hate man
        vo21_mat = DenseMatrix(np.mat([[6,34],[61,94]])) #love boy
        vo22_mat = DenseMatrix(np.mat([[2,10],[17,26]])) #love car
                             
        train_vo_data = [("hate_boy", "man", "man_hate_boy"),
                      ("hate_man", "man", "man_hate_man"),
                      ("hate_boy", "boy", "boy_hate_boy"),
                      ("hate_man", "boy", "boy_hate_man"),
                      ("love_car", "boy", "boy_love_car"),
                      ("love_boy", "man", "man_love_boy"),
                      ("love_boy", "boy", "boy_love_boy"),
                      ("love_car", "man", "man_love_car")
                      ]
        
        # if do not find a phrase
        # what to do?
        train_v_data = [("love", "boy", "love_boy"),
                        ("hate", "man", "hate_man"),
                        ("hate", "boy", "hate_boy"),
                        ("love", "car", "love_car")]
        
        
        sentences = ["man_hate_boy", "car_hate_boy", "boy_hate_boy", 
                     "man_hate_man", "car_hate_man", "boy_hate_man",  
                     "man_love_boy", "car_love_boy", "boy_love_boy",  
                     "man_love_car", "car_love_car", "boy_love_car" ]
        n_mat = DenseMatrix(np.mat([[3,4],[1,2],[5,6]]))
        
        
        n_space = Space(n_mat, ["man", "car", "boy"], self.ft)
        
        s1_mat = (vo11_mat * n_mat.transpose()).transpose()
        s2_mat = (vo12_mat * n_mat.transpose()).transpose()
        s3_mat = (vo21_mat * n_mat.transpose()).transpose()
        s4_mat = (vo22_mat * n_mat.transpose()).transpose()
        
        s_mat = vo11_mat.nary_vstack([s1_mat,s2_mat,s3_mat,s4_mat])
        s_space = Space(s_mat, sentences, self.ft)

        #test train 2d
        model = LexicalFunction(learner=LstsqRegressionLearner(intercept=False))
        vo_space = model.learn_function_space(train_vo_data, n_space, s_space)
        
        self.assertListEqual(vo_space.id2row, ["hate_boy", "hate_man","love_boy", "love_car"])
        self.assertTupleEqual(vo_space.element_shape, (2,2))
        vo11_mat.reshape((1,4))
        np.testing.assert_array_almost_equal(vo11_mat.mat,
                                             vo_space.cooccurrence_matrix.mat[0])
        vo12_mat.reshape((1,4))
        np.testing.assert_array_almost_equal(vo12_mat.mat,
                                             vo_space.cooccurrence_matrix.mat[1])
        vo21_mat.reshape((1,4))
        np.testing.assert_array_almost_equal(vo21_mat.mat,
                                             vo_space.cooccurrence_matrix.mat[2])
        vo22_mat.reshape((1,4))
        np.testing.assert_array_almost_equal(vo22_mat.mat,
                                             vo_space.cooccurrence_matrix.mat[3])
        
        # test train 3d
        model2 = LexicalFunction()
        v_space = model2.learn_function_space(train_v_data, n_space, vo_space)
        np.testing.assert_array_almost_equal(v_mat.mat,
                                             v_space.cooccurrence_matrix.mat)
        self.assertListEqual(v_space.id2row, ["hate","love"])
        self.assertTupleEqual(v_space.element_shape, (2,2,2))
        
        # test compose 3d
        vo_space2 = model2.compose(train_v_data, (v_space,n_space))
        id2row1 = list(vo_space.id2row)
        id2row2 = list(vo_space2.id2row)
        id2row2.sort()
        self.assertListEqual(id2row1, id2row2)
        row_list = vo_space.id2row
        vo_rows1 = vo_space.get_rows(row_list)
        vo_rows2 = vo_space2.get_rows(row_list)
        np.testing.assert_array_almost_equal(vo_rows1.mat, vo_rows2.mat,7)
        self.assertTupleEqual(vo_space.element_shape, vo_space2.element_shape)    
        
           
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