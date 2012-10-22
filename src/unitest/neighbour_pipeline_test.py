'''
Created on Oct 19, 2012

@author: thenghia.pham
'''
import unittest
from unitest import data_dir
from unitest import toolkit_dir
import pipelines.compute_neighbours as find_neighbours 

def read_neighbours_list(file_name):
    result = []
    word = None
    neighbours = []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if (line != ""):
                elements = line.split()
                if (len(elements) == 1):
                    if word != None:
                        result.append((word,neighbours))
                        neighbours = []
                    else:
                        word = elements[0]
                else:
                    neighbours.append((elements[0],elements[1]))
        if word != None:
            result.append((word,neighbours))
    return result 


class NeighboursPipelineTest(unittest.TestCase):


    def setUp(self):
        self.dir_ = data_dir
        self.log_dir = toolkit_dir + "/log/"

    def test_find_neighbours(self):
        '''find_neighbours.main(["compute_neighbours.py", 
                           "-l", self.log_dir + "neighbours_log.txt",
                           "-i", self.dir_ + "neighbours_input.txt",
                           "-m", "dot_prod",
                           "-n", "3",  
                           "-s", self.dir_ + "CORE_SS.mat3.raw.top_sum_3.svd_2.pkl",
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
                               "--space", "%sCORE_SS.mat3.raw.top_sum_3.svd_2.pkl,%sCORE_SS.mat3.raw.top_sum_3.svd_2.pkl" %(self.dir_,self.dir_),
                               "%sconfig/neighbours_config.cfg" %self.dir_
                              ])'''
        find_neighbours.main(["compute_neighbours.py",
                               "-m", "euclidean",
                               "-n", "2",
                               "--space", "%sCORE_SS.mat3.raw.top_sum_3.svd_2.pkl,%sCORE_SS.mat3.raw.top_sum_3.svd_2.pkl" %(self.dir_,self.dir_),
                               "%sconfig/neighbours_config.cfg" %self.dir_
                              ])
        neighbours_list = read_neighbours_list(self.dir_ + "NEIGHBOURS.neighbours_input.txt.euclidean")
        print len(neighbours_list)
        

    def tearDown(self):
        pass


    def testName(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()