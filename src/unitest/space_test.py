'''
Created on Sep 26, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import numpy as np
import unittest
from unitest import data_dir
from composes.semantic_space.operation import ScalingOperation
from composes.transformation.scaling.epmi_weighting import EpmiWeighting
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.dim_reduction.nmf import Nmf
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.semantic_space.space import Space
from composes.similarity.dot_prod import DotProdSimilarity

class Test(unittest.TestCase):

    def setUp(self):
        self.dir_ = data_dir + "/space_test_resources/"
        self.init_test_cases = [(DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"],
                       ["feat1", "feat2"],
                       {"man":1, "car":0},
                       {"feat1":0, "feat2":1},
                       [ScalingOperation(EpmiWeighting())]),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"],
                       [],
                       {"man":1, "car":0},
                       {},
                       [ScalingOperation(EpmiWeighting())])]

        self.m1 = np.array([[1,2,3]])
        self.row1 = ["a"]
        self.row2 = ["a", "b", "c"]
        self.ft1 = ["f1","f2","f3"]
        self.space1 = Space(DenseMatrix(self.m1),self.row1, self.ft1)

        self.x = np.mat([[1,2,3],[2,4,6],[4,675,43]])
        self.us = np.mat([[  2.19272110e+00,   3.03174768e+00],
                               [  4.38544220e+00,   6.06349536e+00],
                               [  6.76369708e+02,  -4.91431927e-02]])
        self.space2 = Space(DenseMatrix(self.x), self.row2, self.ft1)



    def test_init1(self):
        for (m, id2row, id2col, row2id, col2id, ops) in self.init_test_cases:
            space_ = Space(m, id2row, id2col)
            self.assertIs(m, space_.cooccurrence_matrix)
            self.assertIs(id2row, space_.id2row)
            self.assertIs(id2col, space_.id2column)
            self.assertDictEqual(row2id, space_.row2id)
            self.assertDictEqual(col2id, space_.column2id)
            self.assertListEqual([], space_.operations)

    def test_init2(self):
        for (m, id2row, id2col, row2id, col2id, ops) in self.init_test_cases:
            space_ = Space(m, id2row, id2col, row2id, col2id)
            self.assertIs(m, space_.cooccurrence_matrix)
            self.assertIs(id2row, space_.id2row)
            self.assertIs(id2col, space_.id2column)
            self.assertIs(row2id, space_.row2id)
            self.assertIs(col2id, space_.column2id)
            self.assertListEqual([], space_.operations)

    def test_init3(self):
        for (m, id2row, id2col, row2id, col2id, ops) in self.init_test_cases:
            space_ = Space(m, id2row, id2col, operations = ops)
            self.assertIs(m, space_.cooccurrence_matrix)
            self.assertIs(id2row, space_.id2row)
            self.assertIs(id2col, space_.id2column)
            self.assertDictEqual(row2id, space_.row2id)
            self.assertDictEqual(col2id, space_.column2id)
            self.assertListEqual(ops, space_.operations)

    def test_init4(self):
        for (m, id2row, id2col, row2id, col2id, ops) in self.init_test_cases:
            space_ = Space(m, id2row, id2col, row2id, col2id, operations = ops)
            self.assertIs(m, space_.cooccurrence_matrix)
            self.assertIs(id2row, space_.id2row)
            self.assertIs(id2col, space_.id2column)
            self.assertIs(row2id, space_.row2id)
            self.assertIs(col2id, space_.column2id)
            self.assertIs(ops, space_.operations)

    def test_init_raise(self):
        test_cases = [(DenseMatrix(np.array([[1,2],[3,4],[5,6]])),
                       ["car", "man"], ["feat1", "feat2"],
                       {"man":1, "car":0}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       [], ["feat1", "feat2"],
                       {"man":1, "car":0}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1", "feat2"],
                       {}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1"],
                       {"man":1, "car":0}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1"],
                       {"man":1, "car":0}, {"feat1":0, "feat2":1}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1","feat2"],
                       {"man":1, "car":0}, {"feat1":0}),
                      (DenseMatrix(np.array([[1,2],[3,4]])),
                       ["car", "man"], ["feat1","feat2"],
                       {"man":1, "car":0}, {"feat1":1,"feat2":0})
                      ]

        for (m, id2row, id2col, row2id, col2id) in test_cases:
            self.assertRaises(ValueError, Space, m, id2row, id2col,
                              row2id, col2id)

    def test_apply_weighting_operation(self):
        test_cases = [(self.space1, np.array([[1,1,1]]))]
        w = EpmiWeighting()
        for in_s, expected_mat in test_cases:
            out_s = in_s.apply(w)
            np.testing.assert_array_almost_equal(expected_mat,
                                                 out_s.cooccurrence_matrix.mat,
                                                 7)
            self.assertListEqual(out_s.id2row, in_s.id2row)
            self.assertListEqual(out_s.id2column, in_s.id2column)
            self.assertDictEqual(out_s.row2id, in_s.row2id)
            self.assertDictEqual(out_s.column2id, in_s.column2id)
            self.assertEqual(1, len(out_s.operations))

    def test_get_sim(self):
        test_cases = [(self.space2, None, "a", "b", 28),
                      (self.space2, None, "a", "none", 0),
                      (self.space2, self.space2, "a", "b", 28),
                      (self.space2, self.space1, "a", "a", 14),
                      (self.space2, self.space1, "a", "none", 0),
                      (self.space2, self.space1, "none", "a", 0)
                      ]
        for space, space2, w1, w2, expected_sim in test_cases:
            if space2 is None:
                sim = space.get_sim(w1, w2, DotProdSimilarity())
            else:
                sim = space.get_sim(w1, w2, DotProdSimilarity(), space2)

            self.assertEqual(sim, expected_sim)

    def test_get_sim_raises(self):

        space3 = self.space2.apply(Svd(2))
        self.assertRaises(ValueError, self.space2.get_sim,
                          "a", "b", DotProdSimilarity(), space3)

    def test_get_neighbours(self):
        test_cases = [(self.space2, None, "a", 0,[]),
                      (self.space2, None, "a", 1,[("c",1483)]),
                      (self.space2, None, "a", 2,[("c",1483),("b",28)]),
                      (self.space2, None, "a", 3,[("c",1483),("b",28),("a",14)]),
                      (self.space2, None, "a", 4,[("c",1483),("b",28),("a",14)]),
                      (self.space2, self.space1, "a", 1,[("a",14)]),
                      (self.space2, self.space1, "a", 3,[("a",14)])
                      #(self.space2, self.space1, "none", 4,[])
                      ]
        for space, space2, word, no_neigh, expected in test_cases:
            if space2 is None:
                neighbours = space.get_neighbours(word, no_neigh,
                                                  DotProdSimilarity())
            else:
                neighbours = space.get_neighbours(word, no_neigh,
                                                  DotProdSimilarity(),
                                                  space2)

            self.assertListEqual(expected, neighbours)

    def test_get_neighbours_raises(self):

        space3 = self.space2.apply(Svd(2))
        self.assertRaises(ValueError, self.space2.get_neighbours,
                          "a", 2, DotProdSimilarity(), space3)


    def test_vstack(self):
        space3 = Space(DenseMatrix(self.x[0:2,:]), ["e","f"], self.ft1)

        space4 = Space(DenseMatrix(np.vstack((self.x, self.x[0:2,:]))),
                       ["a", "b", "c", "e","f"], self.ft1)

        test_cases = [(self.space2, space3, space4)]
        for space1, space2, expected_space in test_cases:
            outcome = space1.vstack(space1, space2)
            np.testing.assert_array_equal(expected_space.cooccurrence_matrix.mat,
                                          outcome.cooccurrence_matrix.mat)

            self.assertListEqual(outcome.id2column, space1.id2column)
            self.assertListEqual(outcome.id2column, expected_space.id2column)

            self.assertDictEqual(outcome.column2id, space1.column2id)
            self.assertDictEqual(outcome.column2id, expected_space.column2id)

            self.assertListEqual(outcome.id2row, expected_space.id2row)
            self.assertDictEqual(outcome.row2id, expected_space.row2id)

            self.assertListEqual([], outcome.operations)

    def test_vstack_raises(self):

        space3 = Space(DenseMatrix(self.x[0:2,0:1]), ["e","f"], self.ft1[0:1])
        space4 = Space(DenseMatrix(self.x[0:2,:]), ["a","f"], self.ft1)
        space5 = Space(DenseMatrix(self.x[0:2,:]), ["e","f"], [])
        space6 = Space(DenseMatrix(self.x[0:2,:]), ["e","f"], ["f1","f2","f4"])

        test_cases = [(self.space2, space3),
                      (self.space2, space4),
                      (self.space2, space5),
                      (self.space2, space6)
                      ]

        for space1, space2 in test_cases:
            self.assertRaises(ValueError, space1.vstack, space1, space2)


    def test_apply_svd_reduction(self):
        test_cases = [(self.space2, self.us)]
        red = Svd(2)
        for in_s, expected_mat in test_cases:
            out_s = in_s.apply(red)
            np.testing.assert_array_almost_equal(expected_mat,
                                                 out_s.cooccurrence_matrix.mat,
                                                 2)
            self.assertListEqual(out_s.id2row, in_s.id2row)
            self.assertListEqual(out_s.id2column, [])
            self.assertDictEqual(out_s.row2id, in_s.row2id)
            self.assertDictEqual(out_s.column2id, {})
            self.assertEqual(1, len(out_s.operations))

    def test_apply_nmf_reduction(self):
        test_cases = [(self.space2, self.us)]
        red = Nmf(2)

        for in_s, expected_mat in test_cases:
            out_s = in_s.apply(red)
            self.assertListEqual(out_s.id2row, in_s.id2row)
            self.assertListEqual(out_s.id2column, [])
            self.assertDictEqual(out_s.row2id, in_s.row2id)
            self.assertDictEqual(out_s.column2id, {})
            self.assertEqual(1, len(out_s.operations))

    def test_build_data(self):

        test_cases = [("data1",["red", "blue"], ["car", "man"],
                       np.mat([[3,5],[0,10]]), np.mat([[3,5],[0,10]])),
                      ("data2",["red"], ["car"],
                       np.mat([[3]]), np.mat([[3]])),
                      ("data3",["red", "blue"], ["car", "man"],
                       np.mat([[15,0],[0,6]]), np.mat([[5,0],[0,6]])),
                      ("data7",["red"], ["car"], np.mat([[0]]), np.mat([[0]])),
                      ("data9",["man"], ["car"], np.mat([[4]]), None),
                      ]
        for data_file, rows, cols, smat, dmat in test_cases:
            data_file1 = self.dir_ + data_file + ".sparse"

            sp = Space.build(data=data_file1,
                             cols= self.dir_ + data_file + ".cols",
                             format="sm")
            self.assertListEqual(rows, sp.id2row)
            self.assertListEqual(cols, sp.id2column)

            self.assertIsInstance(sp.cooccurrence_matrix, SparseMatrix)
            np.testing.assert_array_equal(smat,
                                          sp.cooccurrence_matrix.mat.todense())

            data_file2 = self.dir_ + data_file + ".dense"
            if not dmat is None:
                sp = Space.build(data=data_file2, format="dm")
                self.assertListEqual(rows, sp.id2row)
                self.assertListEqual([], sp.id2column)

                self.assertIsInstance(sp.cooccurrence_matrix, DenseMatrix)
                np.testing.assert_array_equal(dmat, sp.cooccurrence_matrix.mat)


    def test_build_data_row(self):
        test_cases = [("data1", "row1.row", ["red"], ["car", "man"],
                       np.mat([[3,5]]), np.mat([[3,5]])),
                      ("data2", "row1.row",["red"], ["car"],
                       np.mat([[3]]), np.mat([[3]])),
                      ("data3", "row2.row", ["blue", "red"], ["car", "man"],
                       np.mat([[0,6],[15,0]]), np.mat([[0,6],[5,0]])),
                      ("data3", "row3.row", ["blue", "red"], ["car", "man"],
                       np.mat([[0,6],[15,0]]), np.mat([[0,6],[5,0]])),
                      ("data7", "row2.row", ["blue", "red"], ["car"],
                       np.mat([[0],[0]]), np.mat([[0],[0]])),
                      ]

        for data_file, row_file, rows, cols, smat, dmat in test_cases:
            row_file = self.dir_ + row_file

            data_file1 = self.dir_ + data_file + ".sparse"

            sp = Space.build(data=data_file1,
                             rows= row_file,
                             cols= self.dir_ + data_file + ".cols",
                             format="sm")
            self.assertListEqual(rows, sp.id2row)
            self.assertListEqual(cols, sp.id2column)

            self.assertIsInstance(sp.cooccurrence_matrix, SparseMatrix)
            np.testing.assert_array_equal(smat,
                                          sp.cooccurrence_matrix.mat.todense())

            data_file2 = self.dir_ + data_file + ".dense"

            sp = Space.build(data=data_file2, rows= row_file, format="dm")
            self.assertListEqual(rows, sp.id2row)
            self.assertListEqual([], sp.id2column)

            self.assertIsInstance(sp.cooccurrence_matrix, DenseMatrix)
            np.testing.assert_array_equal(dmat, sp.cooccurrence_matrix.mat)

    def test_build_data_row_col(self):
        test_cases = [("data1", "row1.row", "col1.col", ["red"], ["man", "car"],
                       np.mat([[5,3]]), np.mat([[3,5]])),
                      ("data1", "row1.row", "col5.col", ["red"], ["man", "car"],
                       np.mat([[5,3]]), np.mat([[3,5]])),
                      ("data3", "row2.row", "col2.col", ["blue", "red"], ["car"],
                       np.mat([[0],[15]]), None),
                      ("data2", "row1.row","col1.col", ["red"], ["man","car"],
                       np.mat([[0,3]]), None),
                      ("data3", "row3.row", "col3.col", ["blue", "red"], ["man", "car"],
                       np.mat([[6,0],[0,15]]), np.mat([[0,6],[5,0]])),
                      ("data7", "row2.row", "col3.col", ["blue", "red"], ["man", "car"],
                       np.mat([[0,0],[0,0]]), None),
                      ("data3", "row2.row", "col4.col", ["blue", "red"], ["airplane"],
                       np.mat([[0],[0]]), None)
                      ]

        for data_file, row_file, col_file, rows, cols, smat, dmat in test_cases:
            row_file = self.dir_ + row_file
            col_file = self.dir_ + col_file

            data_file1 = self.dir_ + data_file + ".sparse"

            if smat is None:
                self.assertRaises(ValueError, Space.build, data=data_file1, rows= row_file, cols=col_file, format="sm")

            else:
                sp = Space.build(data=data_file1, rows= row_file, cols=col_file, format="sm")
                self.assertListEqual(rows, sp.id2row)
                self.assertListEqual(cols, sp.id2column)

                self.assertIsInstance(sp.cooccurrence_matrix, SparseMatrix)
                np.testing.assert_array_equal(smat,
                                              sp.cooccurrence_matrix.mat.todense())

            data_file2 = self.dir_ + data_file + ".dense"

            if dmat is None:
                self.assertRaises(ValueError, Space.build, data=data_file2, rows= row_file, cols=col_file, format="dm")

            else:
                sp = Space.build(data=data_file2, rows= row_file, cols=col_file, format="dm")
                self.assertListEqual(rows, sp.id2row)
                self.assertListEqual(cols, sp.id2column)

                self.assertIsInstance(sp.cooccurrence_matrix, DenseMatrix)
                np.testing.assert_array_equal(dmat, sp.cooccurrence_matrix.mat)

    def test_build_raises(self):

        dense_test_cases = ["data10.dense", "data9.dense", "data8.dense"]

        for data_file in dense_test_cases:
            data_file = self.dir_ + data_file
            self.assertRaises(ValueError, Space.build, data=data_file, format="dm")

        sparse_test_cases = ["data8.sparse", "data10.sparse"]

        for data_file in sparse_test_cases:
            data_file = self.dir_ + data_file
            self.assertRaises(ValueError, Space.build, data=data_file, format="sm")

    def reset_export_files(self, out_file):
        with open(out_file + ".dm", "w"):
            pass
        with open(out_file + ".row", "w"):
            pass
        with open(out_file + ".col", "w"):
            pass

    def test_export(self):

        out_file = self.dir_ + "tmp"
        mat1 = np.mat([[1,2],[3,0]])
        mat1row, mat1col = ["a","b"], ["f1","f2"]

        mat2 = np.mat([[0,0]])
        mat2row, mat2col = ["a"], []

        test_cases = [(Space(DenseMatrix(mat1), mat1row, mat1col),
                       Space(SparseMatrix(mat1), mat1row, mat1col)),
                       (Space(DenseMatrix(mat2), mat2row, mat1col),
                       Space(SparseMatrix(mat2), mat2row, mat1col))]

        #3 cases allowed at the moment
        for sp_d, sp_s in test_cases:

            self.reset_export_files(out_file)
            sp_d.export(out_file, format="dm")
            new_sp = Space.build(data=out_file + ".dm",
                                 rows=out_file + ".rows",
                                 cols=out_file + ".cols", format="dm")
            self._test_equal_spaces_dense(sp_d, new_sp)

            self.reset_export_files(out_file)
            sp_d.export(out_file, format="sm")
            new_sp = Space.build(data=out_file + ".sm",
                                 rows=out_file + ".rows",
                                 cols=out_file + ".cols", format="sm")
            self._test_equal_spaces_sparse(sp_s, new_sp)

            self.reset_export_files(out_file)
            sp_s.export(out_file, format="sm")
            new_sp = Space.build(data=out_file + ".sm",
                                 rows=out_file + ".rows",
                                 cols=out_file + ".cols", format="sm")
            self._test_equal_spaces_sparse(sp_s, new_sp)

            self.reset_export_files(out_file)
            sp_s.export(out_file, format="dm")
            new_sp = Space.build(data=out_file + ".dm",
                                 rows=out_file + ".rows",
                                 cols=out_file + ".cols", format="dm")

            self._test_equal_spaces_dense(sp_d, new_sp)

        test_cases = [(Space(DenseMatrix(mat2), mat2row, mat2col),
                       Space(SparseMatrix(mat2), mat2row, mat2col))]

        for sp_d, sp_s in test_cases:

            self.reset_export_files(out_file)
            sp_d.export(out_file, format="dm")
            new_sp = Space.build(data=out_file + ".dm",
                                 rows=out_file + ".rows",
                                 format="dm")
            self._test_equal_spaces_dense(sp_d, new_sp)

            self.reset_export_files(out_file)
            sp_s.export(out_file, format="dm")
            new_sp = Space.build(data=out_file + ".dm",
                                 rows=out_file + ".rows",
                                 format="dm")

            self._test_equal_spaces_dense(sp_d, new_sp)

    def _test_equal_spaces_structs(self, sp, new_sp):
        self.assertListEqual(sp.id2row, new_sp.id2row)
        self.assertListEqual(sp.id2column, new_sp.id2column)
        self.assertDictEqual(sp.row2id, new_sp.row2id)
        self.assertDictEqual(sp.column2id, new_sp.column2id)

    def _test_equal_spaces_dense(self, sp, new_sp):

        self._test_equal_spaces_structs(sp, new_sp)
        np.testing.assert_array_equal(sp.cooccurrence_matrix.mat,
                                      new_sp.cooccurrence_matrix.mat)

    def _test_equal_spaces_sparse(self, sp, new_sp):

        self._test_equal_spaces_structs(sp, new_sp)
        np.testing.assert_array_equal(sp.cooccurrence_matrix.mat.todense(),
                                      new_sp.cooccurrence_matrix.mat.todense())

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()