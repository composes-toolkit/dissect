'''
Created on Oct 19, 2012

@author: thenghia.pham
'''
import unittest
from pipelines import build_core_space as bcs
from unitest import data_dir
from unitest import toolkit_dir
import numpy as np
import pipelines.compute_similarities as sim_pipeline

def read_number_list(file_name, column):
    result = []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if (line != ""):
                elements = line.split()
                if (column >= len(elements)):
                    raise ValueError("Expected line to have at least %d elements: %s" 
                                     %(column + 1,line.strip()))
                result.append(float(elements[column]))
                    
    return result 

class SimilarityPipelineTest(unittest.TestCase):

    def setUp(self):
        self.dir_ = data_dir
        self.log_dir = toolkit_dir + "/log/"
        self.cos = np.array([1.0, 0.585984772383, 1.0, 0.585984772383])
        self.dot_prod = np.array([28.0, 1483.0, 14.0, 2966.0])
        self.euclidean = np.array([0.21089672206, 0.00148105508243, 1.0, 0.00148583482109])
        self.lin = np.array([1.0, 0.995623769303, 1.0, 0.991242564101])
        
                #create the spaces required in the tests
        bcs.main(["build_core_space.py", 
          "-l", self.dir_ + "pipelines_test_resources/log1.txt",
          "-i", self.dir_ + "pipelines_test_resources/mat3",
          "-w", "raw",
          "-s", "top_sum_3",
          "-r", "svd_2", 
          "-o", self.dir_  + "pipelines_test_resources/",
          "--input_format", "dm"
          ])

    def test_compute_sim(self):
        sim_pipeline.main(["compute_similarities.py", 
                           "-l", self.log_dir + "sim_log.txt",
                           "-i", self.dir_ + "pipelines_test_resources/sim_input.txt",
                           "-m", "cos,dot_prod,euclidean,lin", 
                           "-s", self.dir_ + "pipelines_test_resources/CORE_SS.mat3.raw.top_sum_3.svd_2.pkl",
                           "-c", "1,2",
                           "-o", self.dir_ + "pipelines_test_resources/"
                           ])
        self.check_sims()
        
        sim_pipeline.main(["compute_similarities.py", 
                           "-i", self.dir_ + "pipelines_test_resources/sim_input.txt",
                           "--sim_measure", "cos,dot_prod,euclidean,lin", 
                           "--space", self.dir_ + "pipelines_test_resources/CORE_SS.mat3.raw.top_sum_3.svd_2.pkl",
                           "--columns", "1,2",
                           "-o", self.dir_ + "pipelines_test_resources/"
                           ])
        self.check_sims()
        
        sim_pipeline.main(["compute_similarities.py", 
                           "--sim_measure", "cos,dot_prod,euclidean,lin", 
                           "--space", self.dir_ + "pipelines_test_resources/CORE_SS.mat3.raw.top_sum_3.svd_2.pkl",
                           "--columns", "1,2",
                           "-o", self.dir_ + "pipelines_test_resources/",
                           self.dir_ + "config/sim_config.cfg"
                           ])
        self.check_sims()
        
        sim_pipeline.main(["compute_similarities.py", 
                           "--sim_measure", "cos,dot_prod,euclidean,lin", 
                           "--space", "%spipelines_test_resources/CORE_SS.mat3.raw.top_sum_3.svd_2.pkl,%spipelines_test_resources/CORE_SS.mat3.raw.top_sum_3.svd_2.pkl" %(self.dir_,self.dir_),
                           "--columns", "1,2",
                           "-o", self.dir_ + "pipelines_test_resources/",
                           self.dir_ + "config/sim_config.cfg"
                           ])
        self.check_sims()
         
        sim_pipeline.main(["compute_similarities.py", 
                           "--sim_measure", "cos,dot_prod,euclidean,lin", 
                           "--in_dir", "%spipelines_test_resources/" % (self.dir_),
                           "--columns", "1,2",
                           "-o", self.dir_ + "pipelines_test_resources/",
                           self.dir_ + "config/sim_config.cfg"
                           ])
                
        self.check_sims()     
        
    def check_sims(self):
        for sim_measure in "cos,dot_prod,euclidean,lin".split(","):
            result_array = np.array(read_number_list("%spipelines_test_resources/SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.%s"
                                               %(self.dir_,sim_measure), 3))
            gold_array = eval("self.%s"%sim_measure)
            #print result_array
            np.testing.assert_array_almost_equal(result_array, gold_array, 5) 

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()