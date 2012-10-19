'''
Created on Oct 18, 2012

@author: georgianadinu
'''
import unittest
import numpy as np
import unittest
from pipelines import build_peripheral_space as bps
from composes.semantic_space.space import Space
import composes.utils.io_utils as io_utils

class Test(unittest.TestCase):


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_raises(self):
        
        self.assertRaises(SystemExit, bps.main,["build_peripheral_space.py","-h"])
        self.assertRaises(SystemExit, bps.main,["build_peripheral_space.py","-l","-h"])

    def test_simple_sparse(self):
            
        bps.main(["build_peripheral_space.py", 
                  "-l", "/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/log1.txt",
                  "-i", "/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/mat1", 
                  "-o", "/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/",
                  "-c", "/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/CORE_SS.mat1.pickle",
                  "--input_format", "sm",
                  "--output_format", "sm"
                  ])
        
        s1 = Space.build(data="/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/mat1.sm", format = "sm")
        
        #s2 = Space.build(data="/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/PER_SS.mat1.sm", format="sm")
        #s3 = io_utils.load("/home/georgianadinu/work/localtoolkit/toolkit/src/unitest/pipelines_test_resources/CORE_SS.mat1.pickle", Space)
        
        #self._test_equal_spaces_sparse(s1, s2)
        #self._test_equal_spaces_sparse(s1, s3)
                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
"""    
        -i --input <file>: prefix of input file.
    -o --output <dir>: output directory. Space is output in pickle format.
    -c --core <file>: core space file (pickle dump of a space, .pickle expected)
    -l --log <file>: log file. Optional, default ./build_core_space.log
    --input_format: <string>: one of sm(sparse matrix), dm(dense matrix), pickle. 
    --output_format: <string> Additional output format: one of sm(sparse matrix), 
            dm(dense matrix). Optional.
    -h --help : help
    """