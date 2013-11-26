'''
Created on Oct 16, 2012

@author: nghia
'''
import unittest
import numpy as np
from unitest import data_dir
from composes.matrix.dense_matrix import DenseMatrix
from composes.semantic_space.space import Space

from composes.composition.weighted_additive import WeightedAdditive
from composes.composition.full_additive import FullAdditive
from composes.composition.dilation import Dilation
from composes.composition.lexical_function import LexicalFunction
from composes.exception.illegal_state_error import IllegalStateError

class ModelExportingTest(unittest.TestCase):

    def setUp(self):
        self.prefix = data_dir + "output/model"
    def test_weighted_additive(self):

        self.m12 = DenseMatrix(np.mat([[3,1],[9,2]]))
        self.m22 = DenseMatrix(np.mat([[4,3],[2,1]]))
        self.ph2 = DenseMatrix(np.mat([[18,11],[24,7]]))
        self.row = ["a", "b"]
        self.ft = ["f1","f2"]
        self.space1 = Space(DenseMatrix(self.m12), self.row, self.ft)
        self.space2 = Space(DenseMatrix(self.ph2), ["a_a","a_b"], self.ft)
        m = WeightedAdditive()
        m.export(self.prefix + ".add1")
        m.train([("a","a","a_a")], self.space1, self.space2)
        m.export(self.prefix + ".add2")

    def test_full_additive(self):

        self.m12 = DenseMatrix(np.mat([[3,1],[9,2]]))
        self.m22 = DenseMatrix(np.mat([[4,3],[2,1]]))
        self.ph2 = DenseMatrix(np.mat([[18,11],[24,7]]))
        self.row = ["a", "b"]
        self.ft = ["f1","f2"]
        self.space1 = Space(DenseMatrix(self.m12), self.row, self.ft)
        self.space2 = Space(DenseMatrix(self.ph2), ["a_a","a_b"], self.ft)
        m = FullAdditive()
        self.assertRaises(IllegalStateError, m.export,self.prefix + ".full1")
        m.train([("a","b","a_b"),("a","a","a_a")], self.space1, self.space2)

        m.export(self.prefix + ".full2")

    def test_dilation(self):

        self.m12 = DenseMatrix(np.mat([[3,1],[9,2]]))
        self.m22 = DenseMatrix(np.mat([[4,3],[2,1]]))
        self.ph2 = DenseMatrix(np.mat([[18,11],[24,7]]))
        self.row = ["a", "b"]
        self.ft = ["f1","f2"]
        self.space1 = Space(DenseMatrix(self.m12), self.row, self.ft)
        self.space2 = Space(DenseMatrix(self.ph2), ["a_a","a_b"], self.ft)
        m = Dilation()
        m.export(self.prefix + ".dil1")
        m.train([("a","b","a_b")], self.space1, self.space2)
        m.export(self.prefix + ".dil2")

    def test_lexical_function(self):

        self.m12 = DenseMatrix(np.mat([[3,1],[9,2]]))
        self.m22 = DenseMatrix(np.mat([[4,3],[2,1]]))
        self.ph2 = DenseMatrix(np.mat([[18,11],[24,7]]))
        self.row = ["a", "b"]
        self.ft = ["f1","f2"]
        self.space1 = Space(DenseMatrix(self.m12), self.row, self.ft)
        self.space2 = Space(DenseMatrix(self.ph2), ["a_a","a_b"], self.ft)
        m = LexicalFunction()
        m._MIN_SAMPLES = 1
        self.assertRaises(IllegalStateError, m.export, self.prefix + ".lf1")
        m.train([("a","b","a_b"),("a","a","a_a")], self.space1, self.space2)
        m.export(self.prefix + ".lf2")



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_weighted_additive']
    unittest.main()