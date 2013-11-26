'''
Created on Oct 8, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import unittest
import numpy as np
from composes.composition.full_additive import FullAdditive
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.utils.regression_learner import RidgeRegressionLearner
from composes.semantic_space.space import Space

class Test(unittest.TestCase):


    def setUp(self):
        self.m11 = DenseMatrix(np.mat([[3],[9]]))
        self.m21 = DenseMatrix(np.mat([[4],[2]]))
        self.ph1 = DenseMatrix(np.mat([[18],[24]]))

        self.space1 = Space(SparseMatrix(np.mat([[3,9],[4,2]])), ["a", "b"], ["f1","f2"])
        self.space2 = Space(SparseMatrix(np.mat([[7,11]])), ["a_b"], ["f1","f2"])
        self.space3 = Space(SparseMatrix(np.mat([[0,0]])), ["a_b"], ["f1","f2"])

        self.space4 = Space(DenseMatrix(np.mat([[3,9],[4,2]])), ["a", "b"], ["f1","f2"])
        self.space5 = Space(DenseMatrix(np.mat([[7,11]])), ["a_b"], ["f1","f2"])
        self.space6 = Space(DenseMatrix(np.mat([[0,0]])), ["a_b"], ["f1","f2"])
        self.space7 = Space(DenseMatrix(np.mat([[7,11],[7,11]])), ["a_b","a_a"], ["f1","f2"])

    def tearDown(self):
        pass


    def test_space_compose_sparse(self):
        #WHAT TO DO HERE???
        #PARAMTERS ARE GIVEN AS DENSE MATRICES, INPUT DATA AS SPARSE??

        test_cases = [([("a","b","a_b")], self.space1, self.space2, DenseMatrix.identity(2), DenseMatrix.identity(2)),
                      ([("a","b","a_b")], self.space1, self.space3, np.mat([[0,0],[0,0]]), np.mat([[0,0],[0,0]]))
                      ]

        for in_data, arg_space, phrase_space, mat_a, mat_b in test_cases:
            comp_model = FullAdditive(A=mat_a, B=mat_b)
            comp_space = comp_model.compose(in_data, arg_space)

            np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat.todense(),
                                                 phrase_space.cooccurrence_matrix.mat.todense(), 10)

    def test_space_compose_dense(self):

        test_cases = [([("a","b","a_b")], self.space4, self.space5, DenseMatrix.identity(2), DenseMatrix.identity(2)),
                      ([("a","b","a_b")], self.space4, self.space6, np.mat([[0,0],[0,0]]), np.mat([[0,0],[0,0]])),
                      ([("a","b","a_b"),("a","b","a_a")], self.space4, self.space7, DenseMatrix.identity(2), DenseMatrix.identity(2)),
                      ]

        for in_data, arg_space, phrase_space, mat_a, mat_b in test_cases:
            comp_model = FullAdditive(A=mat_a, B=mat_b)
            comp_space = comp_model.compose(in_data, arg_space)

            np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat,
                                                 phrase_space.cooccurrence_matrix.mat, 10)

            self.assertListEqual(comp_space.id2column, [])
            self.assertDictEqual(comp_space.column2id, {})

            self.assertListEqual(comp_space.id2row, phrase_space.id2row)
            self.assertDictEqual(comp_space.row2id, phrase_space.row2id)

            self.assertFalse(comp_model._has_intercept)


    def test_space_train_dense(self):

        test_cases = [([("a","b","a_b")], self.space4, self.space5),
                      ([("a","b","a_b")], self.space4, self.space6),
                      ([("a","b","a_b"),("a","b","a_a")], self.space4, self.space7),
                      ]

        learners = [RidgeRegressionLearner(intercept=False, crossvalidation=False, param=0),
                    LstsqRegressionLearner(intercept=False),
                    LstsqRegressionLearner(intercept=True)]

        for in_data, arg_space, phrase_space in test_cases:
            for learner_ in learners:
                comp_model = FullAdditive(learner=learner_)

                comp_model.train(in_data, arg_space, phrase_space)
                comp_space = comp_model.compose(in_data, arg_space)

                np.testing.assert_array_almost_equal(comp_space.cooccurrence_matrix.mat,
                                                     phrase_space.cooccurrence_matrix.mat, 10)

                self.assertListEqual(comp_space.id2column, phrase_space.id2column)
                self.assertDictEqual(comp_space.column2id, phrase_space.column2id)

                self.assertListEqual(comp_space.id2row, phrase_space.id2row)
                self.assertDictEqual(comp_space.row2id, phrase_space.row2id)

                self.assertEqual(comp_model._has_intercept, learner_._intercept)


    def test_train1(self):
        test_cases = [(self.m11, self.m21, self.ph1, np.mat([[2]]), np.mat([[3]])),
                      (self.m11, self.m21, DenseMatrix(np.mat([[0],[0]])),
                       np.mat([[0]]), np.mat([[0]]))]

        for m1, m2, ph, expected_a, expected_b in test_cases:
            comp_model = FullAdditive(learner=LstsqRegressionLearner(intercept=False))
            comp_model._train(m1, m2, ph)
            np.testing.assert_array_almost_equal(comp_model._mat_a_t.transpose().mat,
                                                 expected_a, 10)
            np.testing.assert_array_almost_equal(comp_model._mat_b_t.transpose().mat,
                                                 expected_b, 10)


    def test_train2(self):
        dim_ = 2
        dim_1 = 3
        dim_2 = 5
        for dim in [dim_1 + dim_2, dim_1 + dim_2 + 2]:
            expected_a = np.mat(np.random.random((dim_,dim_1)))
            expected_b = np.mat(np.random.random((dim_,dim_2)))
            m1 = np.mat(np.random.random((dim,dim_1)))
            m2 = np.mat(np.random.random((dim,dim_2)))

            ph = np.mat(expected_a*m1.T + expected_b*m2.T)

            comp_model = FullAdditive(learner=LstsqRegressionLearner(intercept=False))
            comp_model._train(DenseMatrix(m1),DenseMatrix(m2),
                                       DenseMatrix(ph).transpose())
            np.testing.assert_array_almost_equal(comp_model._mat_a_t.transpose().mat,
                                                 expected_a, 10)
            np.testing.assert_array_almost_equal(comp_model._mat_b_t.transpose().mat,
                                                 expected_b, 10)

        for dim in [dim_1 + dim_2 + 6, dim_1 + dim_2 + 20]:
            expected_a = np.mat(np.random.random((dim_,dim_1)))
            expected_b = np.mat(np.random.random((dim_,dim_2)))
            m1 = np.mat(np.random.random((dim,dim_1)))
            m2 = np.mat(np.random.random((dim,dim_2)))

            ph = np.mat(expected_a*m1.T + expected_b*m2.T)

            comp_model = FullAdditive(learner=LstsqRegressionLearner(intercept=True))
            comp_model._train(DenseMatrix(m1),DenseMatrix(m2),
                                       DenseMatrix(ph).transpose())
            np.testing.assert_array_almost_equal(comp_model._mat_a_t.transpose().mat,
                                                 expected_a, 10)
            np.testing.assert_array_almost_equal(comp_model._mat_b_t[:-1,:].transpose().mat,
                                                 expected_b, 10)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()