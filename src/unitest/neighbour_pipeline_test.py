'''
Created on Oct 19, 2012

@author: thenghia.pham
'''
import unittest
import pipelines.compute_neighbours as find_neighbours 

def read_neighbour_list(file_name):
    pass


class NeighboursPipelineTest(unittest.TestCase):


    def setUp(self):
        self.dir_ = "/home/thenghia.pham/git/toolkit/resource/unittest/"
        self.log_dir = "/home/thenghia.pham/git/toolkit/log/"

    def test_find_neighbours(self):
        find_neighbours.main(["compute_neighbours.py", 
                           "-l", self.log_dir + "neighbours_log.txt",
                           "-i", self.dir_ + "neighbours_input.txt",
                           "-m", "dot_prod",
                           "-n", "3",  
                           "-s", self.dir_ + "CORE_SS.mat3.raw.top_sum_3.svd_2.pickle",
                           "-o", self.dir_
                           ])
        
        find_neighbours.main(["compute_neighbours.py", 
                             "%sconfig/neighbours_config.cfg" %self.dir_
                              ])
        
        find_neighbours.main(["compute_neighbours.py",
                               "-m", "lin",
                               "%sconfig/neighbours_config.cfg" %self.dir_
                              ])
        find_neighbours.main(["compute_neighbours.py",
                               "-m", "euclidean",
                               "%sconfig/neighbours_config.cfg" %self.dir_
                              ])
        find_neighbours.main(["compute_neighbours.py",
                               "-m", "euclidean",
                               "--space", "%sCORE_SS.mat3.raw.top_sum_3.svd_2.pickle,%sCORE_SS.mat3.raw.top_sum_3.svd_2.pickle" %(self.dir_,self.dir_),
                               "%sconfig/neighbours_config.cfg" %self.dir_
                              ])
        

    def tearDown(self):
        pass


    def testName(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()