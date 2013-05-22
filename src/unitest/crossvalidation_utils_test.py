'''
Created on Oct 9, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import unittest
from composes.utils.crossvalidation_utils import get_split_indices

class Test(unittest.TestCase):


    def test_get_split_indicec(self):
        
        test_cases = [(10, 3, 4), (9, 10, 1), (10, 10, 1), (109, 10, 11), (1, 1, 1)]

        for range_, fold, max_len in test_cases:

            indices = get_split_indices(range_, fold)
            self.assertGreaterEqual(fold, len(indices))
            
            for i in range(len(indices)):
                self.assertTrue(len(indices[i]) >= range_//fold or fold >= range_)
                self.assertGreaterEqual(max_len, len(indices[i]))
                
        
        indices = get_split_indices(10, 3)
        self.assertEqual(len(indices[0]), 4)
        self.assertEqual(len(indices[1]), 3)
        self.assertEqual(len(indices[2]), 3)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_get_split_indicec']
    unittest.main()