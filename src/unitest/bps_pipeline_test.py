'''
Created on Oct 18, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
from pipelines import build_peripheral_space as bps
from pipelines import build_core_space as bcs
from composes.semantic_space.space import Space

class Test(unittest.TestCase):


    def setUp(self):
        self.dir_ = "/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/"

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
                                      new_sp.cooccurrence_matrix.mat, 6)
            
    def _test_equal_spaces_sparse(self, sp, new_sp):

        self._test_equal_spaces_structs(sp, new_sp)
        np.testing.assert_array_almost_equal(sp.cooccurrence_matrix.mat.todense(), 
                                      new_sp.cooccurrence_matrix.mat.todense(), 6)
        
    def test_raises(self):
        
        self.assertRaises(SystemExit, bps.main,["build_peripheral_space.py","-h"])
        self.assertRaises(SystemExit, bps.main,["build_peripheral_space.py","-l","-h"])

    def test_simple_sparse(self):
            
        bps.main(["build_peripheral_space.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "mat1", 
                  "-o", self.dir_,
                  "-c", self.dir_ + "CORE_SS.mat1.pickle",
                  "--input_format", "sm",
                  "--output_format", "sm"
                  ])
        
        s1 = Space.build(data=self.dir_ + "mat1.sm", format = "sm")
        s2 = Space.build(data=self.dir_ + "PER_SS.mat1.CORE_SS.mat1.sm", format="sm")
        
        self._test_equal_spaces_sparse(s1, s2)
    
    def test_simple_dense(self):    
        bps.main(["build_peripheral_space.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "mat2", 
                  "-o", self.dir_,
                  "-c", self.dir_ + "CORE_SS.mat2.pickle",
                  "--input_format", "dm",
                  "--output_format", "dm"
                  ])
        s1 = Space.build(data=self.dir_ + "mat2.dm", format = "dm")
        s2 = Space.build(data=self.dir_ + "PER_SS.mat2.CORE_SS.mat2.dm", format="dm")
        
        self._test_equal_spaces_dense(s1, s2)
        
    def test_simple_ops(self):

        bcs.main(["build_core_space.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "mat3",
                  "-w", "raw",
                  "-s", "top_sum_3,top_length_3,top_sum_4",
                  "-r", "svd_2,svd_1",
                  "-o", self.dir_,
                  "--input_format", "dm",
                  "--output_format", "dm"
                  ])        
        
        core_mats = ["CORE_SS.mat3.raw.top_sum_3.svd_2",
                     "CORE_SS.mat3.raw.top_sum_3.svd_1",
                     "CORE_SS.mat3.raw.top_length_3.svd_2",
                     "CORE_SS.mat3.raw.top_length_3.svd_1",
                     "CORE_SS.mat3.raw.top_sum_4.svd_2",
                     "CORE_SS.mat3.raw.top_sum_4.svd_1"
                     ]
        
        core_spaces = [Space.build(data=self.dir_ + suffix + ".dm", format="dm") for suffix in core_mats] 
           
        for i, core_mat in enumerate(core_mats):
            bps.main(["build_peripheral_space.py", 
                      "-l", self.dir_ + "log1.txt",
                      "-i", self.dir_ + "mat3", 
                      "-o", self.dir_ ,
                      "-c", self.dir_ + core_mat + ".pickle",
                      "--input_format", "dm",
                      "--output_format", "dm"
                      ])
        
            s1 = core_spaces[i]
            data_file = self.dir_ + "PER_SS.mat3." + core_mats[i] + ".dm"
            s2 = Space.build(data=data_file, format="dm")
            self._test_equal_spaces_dense(s1, s2) 
        
            bps.main(["build_peripheral_space.py", 
                      "-l", self.dir_ + "log1.txt",
                      "-i", self.dir_ + "mat3", 
                      "-o", self.dir_ ,
                      "-c", self.dir_ + core_mat + ".pickle",
                      "--input_format", "sm",
                      "--output_format", "dm"
                      ])
        
            s1 = core_spaces[i]
            data_file = self.dir_ + "PER_SS.mat3." + core_mats[i] + ".dm"
            s2 = Space.build(data=data_file, format="dm")
            
            self._test_equal_spaces_dense(s1, s2) 
        
   
                   
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    