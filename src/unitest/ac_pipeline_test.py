'''
Created on Oct 19, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from unitest import data_dir
import pipelines.train_composition as tc
import pipelines.apply_composition as ac 
import pipelines.build_core_space as bcs
from composes.semantic_space.space import Space
from composes.utils import io_utils

class Test(unittest.TestCase):


    def setUp(self):
        
        self.dir_ = data_dir + "pipelines_test_resources/"
        
        #use as a conversion tool, creates the files we want
        bcs.main(["build_core_space.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "N_mat", 
                  "-o", self.dir_,
                  "--input_format", "dm"
                  ])
        
        bcs.main(["build_core_space.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "AN_mat", 
                  "-o", self.dir_,
                  "--input_format", "dm"
                  ])
 
        bcs.main(["build_core_space.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "A_mat", 
                  "-o", self.dir_,
                  "--input_format", "dm"
                  ])
                        
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
        

    def tearDown(self):
        pass

    def _test_equal_spaces_structs(self, sp, new_sp):     
        self.assertListEqual(sp.id2row, new_sp.id2row)
        self.assertListEqual(sp.id2column, new_sp.id2column)
        self.assertDictEqual(sp.row2id, new_sp.row2id)
        self.assertDictEqual(sp.column2id, new_sp.column2id)
        
    def _test_equal_spaces_dense(self, sp, new_sp):

        self._test_equal_spaces_structs(sp, new_sp)
        np.testing.assert_array_almost_equal(sp.cooccurrence_matrix.mat, 
                                      new_sp.cooccurrence_matrix.mat)
            
    def _test_equal_spaces_sparse(self, sp, new_sp):

        self._test_equal_spaces_structs(sp, new_sp)
        np.testing.assert_array_almost_equal(sp.cooccurrence_matrix.mat.todense(), 
                                      new_sp.cooccurrence_matrix.mat.todense())

    def test_simple_load(self):
        
        #trained = io_utils.load(self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.pickle")
        #new_space = trained.function_space
        
        ac.main(["apply_composition.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "an_train_data.txt", 
                  "-o", self.dir_,
                  "-t", self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.pickle",
                  "-a", self.dir_ + "CORE_SS.N_mat.pickle",
                  "--output_format", "dm"
                  ]
                )
        
        sp1 = Space.build(data=self.dir_ + "COMPOSED_SS.LexicalFunction.an_train_data.txt.dm",
                         format="dm")

        sp2 = Space.build(data=self.dir_ + "AN_mat.dm",
                         format="dm")

        self._test_equal_spaces_dense(sp1, sp2)

    def test_simple_define(self):
            
        #trained = io_utils.load(self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.pickle")
        #new_space = trained.function_space

        #compose with lexical function
        ac.main(["apply_composition.py", 
          "-l", self.dir_ + "log1.txt",
          "-i", self.dir_ + "an_train_data.txt", 
          "-o", self.dir_,
          "-t", self.dir_ + "TRAINED_COMP_MODEL.lexical_func.an_train_data.txt.pickle",
          "-a", self.dir_ + "CORE_SS.N_mat.pickle",
          "--output_format", "dm"
          ]
        )
        
        sp2 = Space.build(data=self.dir_ + "COMPOSED_SS.LexicalFunction.an_train_data.txt.dm",
                         format="dm")
        
        #compose with weighted addition        
        ac.main(["apply_composition.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "an_train_data.txt", 
                  "-o", self.dir_,
                  "-m", "weighted_add",
                  "--alpha", "0.5",
                  "--beta", "0.5",
                  "-a", self.dir_ + "CORE_SS.A_mat.pickle"+ "," + self.dir_ + "CORE_SS.N_mat.pickle",
                  "--output_format", "dm"
                  ]
                )
        
        sp1 = Space.build(data=self.dir_ + "COMPOSED_SS.WeightedAdditive.an_train_data.txt.dm",
                         format="dm")
        sp3 = io_utils.load(self.dir_ + "COMPOSED_SS.WeightedAdditive.an_train_data.txt.pickle")
        
        np.testing.assert_array_equal(sp1.cooccurrence_matrix.mat, np.mat([[3,4],[4,5]]))
        self._test_equal_spaces_structs(sp1, sp2)
        sp1.to_sparse()
        sp3.to_sparse()
        self._test_equal_spaces_sparse(sp1, sp3)
        
        #the two output format have to contain identical data
        sp1.to_dense()
        sp3.to_dense()
        self._test_equal_spaces_dense(sp1, sp3)
        
        #compose with dilation 
        ac.main(["apply_composition.py", 
          "-l", self.dir_ + "log1.txt",
          "-i", self.dir_ + "an_train_data.txt", 
          "-o", self.dir_,
          "-m", "dilation",
          "--lambda", "1",
          "-a", self.dir_ + "CORE_SS.A_mat.pickle"+ "," + self.dir_ + "CORE_SS.N_mat.pickle",
          "--output_format", "dm"
          ]
        )
        
        sp1 = Space.build(data=self.dir_ + "COMPOSED_SS.Dilation.an_train_data.txt.dm",
                         format="dm")
        n_space = io_utils.load(self.dir_ + "CORE_SS.N_mat.pickle")
        sp1.to_dense()
        n_space.to_dense()
        np.testing.assert_array_almost_equal(sp1.cooccurrence_matrix.mat, n_space.cooccurrence_matrix.mat)
        self._test_equal_spaces_structs(sp1, sp2)
        
        #compose with dilation, change the order of the arguments
        ac.main(["apply_composition.py", 
          "-l", self.dir_ + "log1.txt",
          "-i", self.dir_ + "na_train_data.txt", 
          "-o", self.dir_,
          "-m", "dilation",
          "--lambda", "1",
          "-a", self.dir_ + "CORE_SS.N_mat.pickle" + "," + self.dir_ + "CORE_SS.A_mat.pickle",
          "--output_format", "dm"
          ]
        )
        
        sp1 = Space.build(data=self.dir_ + "COMPOSED_SS.Dilation.na_train_data.txt.dm",
                         format="dm")

        sp1.to_dense()
        np.testing.assert_array_almost_equal(sp1.cooccurrence_matrix.mat, np.mat([[3,4],[3,4]]),5)
        self._test_equal_spaces_structs(sp1, sp2)
 
        #compose with multiplicative
        ac.main(["apply_composition.py", 
          "-l", self.dir_ + "log1.txt",
          "-i", self.dir_ + "aan_train_data.txt", 
          "-o", self.dir_,
          "-m", "mult",
          "-a", self.dir_ + "CORE_SS.A_mat.pickle"+ "," + self.dir_ + "COMPOSED_SS.Dilation.an_train_data.txt.pickle",
          "--output_format", "dm"
          ]
        )
        
        sp1 = Space.build(data=self.dir_ + "COMPOSED_SS.Multiplicative.aan_train_data.txt.dm",
                         format="dm")

        #sp1.to_dense()
        #np.testing.assert_array_almost_equal(sp1.cooccurrence_matrix.mat, np.mat([[9,16],[16,25]]),5)
        #self._test_equal_spaces_structs(sp1, sp2)       
                       
               
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()