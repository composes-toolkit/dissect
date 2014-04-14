'''
Created on Oct 9, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import unittest
import numpy as np
from composes.utils.regression_learner import RidgeRegressionLearner
from composes.utils.regression_learner import LstsqRegressionLearner
from composes.matrix.dense_matrix import DenseMatrix
from composes.utils.matrix_utils import padd_matrix

class Test(unittest.TestCase):


    def test_trivial_crossvalidation(self):

        for i in range(1, 10):
            m_a = DenseMatrix(np.mat(np.random.random((i + 1,4))))
            m_b = DenseMatrix(np.mat(np.random.random((i + 1,4))))
            tmp_a = m_a.mat.copy()
            tmp_b = m_b.mat.copy()

            learner = RidgeRegressionLearner(param_range=[3], intercept=False)
            solution = learner.train(m_a, m_b)

            learner2 = RidgeRegressionLearner(param = 3, intercept=False)
            solution2 = learner2.train(m_a, m_b)

            np.testing.assert_array_equal(tmp_a, m_a.mat)
            np.testing.assert_array_equal(tmp_b, m_b.mat)
            np.testing.assert_array_equal(solution.mat, solution2.mat)

            learner = RidgeRegressionLearner(param_range=[3], intercept=False)
            solution = learner.train(m_a, m_b)

            np.testing.assert_array_equal(tmp_a, m_a.mat)
            np.testing.assert_array_equal(tmp_b, m_b.mat)
            np.testing.assert_array_equal(solution.mat, solution2.mat)

            learner = RidgeRegressionLearner(param_range=[0], intercept=False)
            solution = learner.train(m_a, m_b)

            learner2 = LstsqRegressionLearner(intercept=False)
            solution2 = learner2.train(m_a, m_b)

            np.testing.assert_array_almost_equal(solution.mat, solution2.mat, 3)


    def test_crossvalidation(self):

        a = DenseMatrix(np.matrix([[1, 1],[2, 3],[4, 6]]))
        b = DenseMatrix(np.matrix([[12, 15, 18],[21, 27, 33],[35, 46, 57]]))
        res = DenseMatrix(np.matrix([[1, 2, 3],[4, 5, 6],[7, 8, 9]]))

        learner = RidgeRegressionLearner(intercept=True, param_range=[0])
        learner2 = LstsqRegressionLearner(intercept=False)

        res1 = learner2.train(a, b)
        res2 = learner.train(a, b)

        np.testing.assert_array_almost_equal(res2.mat[:-1,:], res[0:2,:].mat, 6)
        np.testing.assert_array_almost_equal(res2.mat[-1,:], res[2:3,:].mat, 6)

        new_a = padd_matrix(a, 1)
        self.assertGreater(((a * res1) - b).norm(), ((new_a * res2) - b).norm())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_trivial_cases']
    unittest.main()