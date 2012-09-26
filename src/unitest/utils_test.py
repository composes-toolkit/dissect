'''
Created on Sep 26, 2012

@author: georgianadinu
'''
import unittest
from composes.utils.space_utils import list2dict 

class UtilsTest(unittest.TestCase):


    def test_list2dict(self):
        test_cases = [(["a","v","d"], {"a":0, "v":1, "d":2}), ([], {})]
        
        for list_, expected in test_cases:
            outcome = list2dict(list_)
            self.assertDictEqual(outcome, expected)

        self.assertRaises(ValueError, list2dict, ["a","v","a"]) 
           

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()