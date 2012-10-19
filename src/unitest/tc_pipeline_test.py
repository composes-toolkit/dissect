'''
Created on Oct 19, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
import pipelines.train_composition as tc 
import pipelines.build_core_space as bcs
from composes.utils import io_utils

class Test(unittest.TestCase):


    def setUp(self):
        self.dir_ = "/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/"


    def tearDown(self):
        pass


    def test_simple(self):
        
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
                
        tc.main(["train_composition.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "an_train_data.txt", 
                  "-o", self.dir_,
                  "-m", "lexical_func",
                  "-p", self.dir_ + "CORE_SS.AN_mat.pickle",
                  "-a", self.dir_ + "CORE_SS.N_mat.pickle",
                  "-r", "lstsq",
                  "--intercept", True,
                  "--export_params", True,
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

        
        

#TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.params.row


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
