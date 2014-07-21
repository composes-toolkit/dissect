'''
Created on Oct 16, 2012

@author: nghia
'''
import logging

import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import gzip as gzip
from warnings import warn
from scipy.sparse import csr_matrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.utils.gen_utils import assert_is_instance
import struct

logger = logging.getLogger(__name__)


def save(object_, file_name):
    create_parent_directories(file_name)
    try:
        with open(file_name, 'w') as f:
            pickle.dump(object_, f, 2)
    except struct.error:
        warn("object is too big, using pickle with protocol 0")
        with open(file_name, 'w') as f:
            pickle.dump(object_, f, 0)


def load(file_name, data_type=None):
    with open(file_name) as f:
        result = pickle.load(f)

    if not data_type is None:
        assert_is_instance(result, data_type)

    return result


def create_directories(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_parent_directories(file_name):
    parent_dir = os.path.dirname(file_name)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def extract_indexing_structs(filename, field_list):
    str2id = {}
    id2str = []
    no_fields = len(field_list)

    str2id_list = [str2id.copy() for i in xrange(no_fields)]
    id2str_list = [list(id2str) for i in xrange(no_fields)]
    index_list = [0 for i in xrange(no_fields)]
    max_field = max(field_list)

    if filename.endswith(".gz"):
        input_stream = gzip.open(filename, "rb")
    else:
        input_stream = open(filename, "rb")

    for line in input_stream:
        if line.strip() != "":
            elements = line.strip().split()
            if len(elements) <= max_field:
                warn("Invalid input line:%s. Skipping it" % line.strip())
            else:
                for field_idx, field in enumerate(field_list):
                    current_str = elements[field]
                    if not current_str in str2id_list[field_idx]:
                        str2id_list[field_idx][current_str] = index_list[field_idx]
                        id2str_list[field_idx].append(current_str)
                        index_list[field_idx] += 1

    for id2str in id2str_list:
        if not id2str:
            raise ValueError("Found no valid data in file: %s!" % filename)
    return (id2str_list, str2id_list)


def read_tuple_list(data_file, fields=None):
    field_list = []
    result = []

    if fields:
        field_list = fields

    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line != "":
                elements = line.split()
                if field_list:
                    try:
                        elements = np.array(elements)[field_list]
                    except IndexError:
                        raise IndexError("Cannot extract fields:%s from %s!"
                                         % (field_list, data_file))

                result.append(tuple(elements))

    return result


def read_list(file_name, **kwargs):
    field = None
    result = []
    if "field" in kwargs:
        field = kwargs["field"]

    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if line != "":
                if not field is None:
                    try:
                        result.append(line.split()[field])
                    except IndexError:
                        raise IndexError("Cannot extract field:%s from %s!"
                                         % (field, file_name))
                else:
                    result.append(line)
    return result


def read_sparse_space_data(matrix_file, row2id, column2id, dtype=np.double):
    logger.info('Loading sparse space from %s', matrix_file)
    if matrix_file.endswith(".gz"):
        f = gzip.open(matrix_file, "rb")
    else:
        f = open(matrix_file, "rb")

    no_lines = sum(1 for line in f if line.strip() != "")
    f.close()

    row = np.zeros(no_lines, dtype=np.int32)
    col = np.zeros(no_lines, dtype=np.int32)

    data = np.zeros(no_lines, dtype=dtype)

    if matrix_file.endswith(".gz"):
        f = gzip.open(matrix_file, "rb")
    else:
        f = open(matrix_file, "rb")

    i = 0
    for line in f:
        if line.strip() != "":
            line_elements = line.strip().split()
            if len(line_elements) >= 3:
                [word1, word2, count] = line_elements[0:3]
                if word1 in row2id and word2 in column2id:
                    row[i] = row2id[word1]
                    col[i] = column2id[word2]
                    data[i] = dtype(count)
                    i += 1
                    if i % 1000000 == 0:
                        logger.info("Progress... %d", i)
            #if len(line_elements) > 3:
            #    warn("Invalid input line:%s. Expected 3 fields, ignoring additional ones!" % line.strip())
            else:
                raise ValueError("Invalid row: %s, expected at least %d fields"
                                 % (line.strip(), 3))

    f.close()
    # eliminate the extra zeros created when word1 or word2 is not row2id or col2id!!
    data = data[0:i]
    row = row[0:i]
    col = col[0:i]

    m = SparseMatrix(csr_matrix((data, (row, col)), shape=(len(row2id), len(column2id))))
    if m.mat.nnz != i:
        warn("Found 0-counts or duplicate row,column pairs. (Duplicate entries are summed up.)")

    return m


def read_dense_space_data(matrix_file, row2id, element_type=np.double):
    logger.info('Loading dense space from %s', matrix_file)
    #get number of rows and columns
    if matrix_file.endswith(".gz"):
        f = gzip.open(matrix_file, "rb")
    else:
        f = open(matrix_file, "rb")

    first_line = f.next()
    no_cols = len(first_line.strip().split()) - 1
    if no_cols <= 0:
        raise ValueError("Invalid row: %s, expected at least %d fields" % (first_line.strip(), 2))
    f.close()

    no_rows = len(row2id)
    row_string_set = set([])

    m = np.mat(np.zeros(shape=(no_rows, no_cols), dtype=element_type))

    if matrix_file.endswith(".gz"):
        f = gzip.open(matrix_file, "rb")
    else:
        f = open(matrix_file, "rb")

    for line in f:
        if not line.strip() == "":
            elements = line.strip().split()
            if len(elements) != no_cols + 1:
                raise ValueError("Invalid row: %s, expected %d fields"
                                 % (line.strip(), no_cols + 1))
            word = elements[0]
            if word in row2id:
                i = row2id[word]
                if word in row_string_set != 0:
                    warn("Found duplicate row: %s. Ignoring it." % word)
                else:
                    m[i, :] = elements[1:]
                    row_string_set.add(word)

    f.close()

    return DenseMatrix(m)


def print_list(list_, file_name):
    with open(file_name, 'w') as f:
        for item in list_:
            f.write(item + "\n")


def print_cooc_mat_sparse_format(matrix_, id2row, id2column, file_prefix):
    matrix_file = "%s.%s" % (file_prefix, "sm")
    if not id2column:
        raise ValueError("Cannot print matrix with no column info in sparse format!")

    mat = matrix_.mat
    with open(matrix_file, 'w') as f:
        if isinstance(matrix_, SparseMatrix):

            data = mat.data
            row_indices = mat.indptr
            col_indices = mat.indices

            row_index = 0
            next_row = row_indices[1]
            row = id2row[0]
            for i in xrange(len(data)):
                while i == next_row:
                    row_index += 1
                    next_row = row_indices[row_index + 1]
                    row = id2row[row_index]
                col = id2column[col_indices[i]]
                f.write("%s\t%s\t%f\n" % (row, col, data[i]))
        else:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if mat[i, j] != 0:
                        f.write("%s\t%s\t%f\n" % (id2row[i], id2column[j], mat[i, j]))


def print_cooc_mat_dense_format(matrix_, id2row, file_prefix):
    matrix_file = "%s.%s" % (file_prefix, "dm")

    with open(matrix_file, 'w') as f:
        for i, row in enumerate(id2row):
            v = DenseMatrix(matrix_[i]).mat.flat
            line = "\t".join([row] + [repr(v[j]) for j in range(len(v))])
            f.write("%s\n" % (line))

