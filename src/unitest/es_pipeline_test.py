'''
Created on Oct 19, 2012

@author: georgianadinu
'''
import unittest
from unitest import data_dir
from pipelines import evaluate_similarities as es

class Test(unittest.TestCase):


    def setUp(self):
        self.dir_ = data_dir + "pipelines_test_resources/"


    def tearDown(self):
        pass


    def test_simple(self):
        
        es.main(["evaluate_similarities.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "pred1.txt",
                  "-m", "pearson,spearman",
                  "-c", "1,3", 
                  ])        

        es.main(["evaluate_similarities.py", 
                  "-l", self.dir_ + "log1.txt",
                  "-i", self.dir_ + "pred1.txt",
                  "--in_dir", self.dir_,
                  "--filter", "pred",
                  "-m", "pearson,spearman",
                  "-c", "1,3", 
                  ]) 
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    