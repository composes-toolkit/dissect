'''
Created on Oct 19, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from unitest import data_dir
import pipelines.train_composition as tc 
import pipelines.build_core_space as bcs
from composes.utils import io_utils
from composes.semantic_space.space import Space

class Test(unittest.TestCase):


    def setUp(self):
        self.dir_ = data_dir + "pipelines_test_resources/"
        
        #use as a conversion tool, creates the files we want
        bcs.main(["build_core_space.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "N_mat", 
                  "-o", self.dir_,
                  "--input_format", "dm",
                  ])
        
        bcs.main(["build_core_space.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "AN_mat", 
                  "-o", self.dir_,
                  "--input_format", "dm",
                  ])

    def tearDown(self):
        pass

    def _test_equal_spaces_structs(self, sp, new_sp):     
        self.assertListEqual(sp.id2row, new_sp.id2row)
        self.assertListEqual(sp.id2column, new_sp.id2column)
        self.assertDictEqual(sp.row2id, new_sp.row2id)
        self.assertDictEqual(sp.column2id, new_sp.column2id)
        
    def _test_equal_spaces_dense(self, sp, new_sp):

        self._test_equal_spaces_structs(sp, new_sp)
        np.testing.assert_array_equal(sp.cooccurrence_matrix.mat, 
                                      new_sp.cooccurrence_matrix.mat)
            
    def _test_equal_spaces_sparse(self, sp, new_sp):

        self._test_equal_spaces_structs(sp, new_sp)
        np.testing.assert_array_equal(sp.cooccurrence_matrix.mat.todense(), 
                                      new_sp.cooccurrence_matrix.mat.todense())
        
    def test_simple_lstsq_inter(self):

        tc.main(["train_composition.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "an_train_data.txt", 
                  "-o", self.dir_,
                  "-m", "lexical_func",
                  "-p", self.dir_ + "CORE_SS.AN_mat.pickle",
                  "-a", self.dir_ + "CORE_SS.N_mat.pickle",
                  "-r", "lstsq",
                  "--intercept", "True",
                  "--export_params", "True",
                  ])        

        trained = io_utils.load(self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.pickle")
        new_space = trained.function_space
        
        np.testing.assert_array_almost_equal(new_space.cooccurrence_matrix.mat, 
                                             np.mat([[0.66666667,0.33333333,
                                                      -0.33333333,0.33333333,
                                                      0.66666667,0.33333333]]),
                                              7)
        
        self.assertTupleEqual(new_space.element_shape, (2,3))
        self.assertListEqual(new_space.id2row, ["big"])
        self.assertListEqual(new_space.id2column, [])

        
        a_space = Space.build(data=self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.params.dm", 
                              format="dm")
        
        self._test_equal_spaces_dense(a_space, new_space)


    def test_simple_lstsq_no_inter(self):
        tc.main(["train_composition.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "an_train_data.txt", 
                  "-o", self.dir_,
                  "-m", "lexical_func",
                  "-p", self.dir_ + "CORE_SS.AN_mat.pickle",
                  "-a", self.dir_ + "CORE_SS.N_mat.pickle",
                  "-r", "lstsq",
                  "--intercept", "False",
                  "--export_params", "True"
                  ]) 
        
        trained = io_utils.load(self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.pickle")
        new_space = trained.function_space
        np.testing.assert_array_almost_equal(new_space.cooccurrence_matrix.mat, 
                                             np.mat([1,0,0,1]), 10)
        self.assertTupleEqual(new_space.element_shape, (2,2))
        self.assertListEqual(new_space.id2row, ["big"])
        self.assertListEqual(new_space.id2column, [])
        
        a_space = Space.build(data=self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.params.dm", 
                              format="dm")
        
        self._test_equal_spaces_dense(a_space, new_space)
        
        tc.main(["train_composition.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "an_train_data.txt", 
                  "-o", self.dir_,
                  "-m", "lexical_func",
                  "-p", self.dir_ + "CORE_SS.AN_mat.pickle",
                  "-a", self.dir_ + "CORE_SS.N_mat.pickle",
                  "-r", "ridge",
                  "--param", "0",
                  "--crossvalidation", "False",
                  "--intercept", "False",
                  "--export_params", "True"
                  ]) 
        
        trained = io_utils.load(self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.pickle")
        new_space2 = trained.function_space
        np.testing.assert_array_almost_equal(new_space2.cooccurrence_matrix.mat, 
                                             np.mat([1,0,0,1]), 10)
        self.assertTupleEqual(new_space2.element_shape, (2,2))
        self.assertListEqual(new_space2.id2row, ["big"])
        self.assertListEqual(new_space2.id2column, [])
        
        a_space = Space.build(data=self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.params.dm", 
                              format="dm")
        
        self._test_equal_spaces_dense(a_space, new_space2)        
         
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
