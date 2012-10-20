'''
Created on Sep 26, 2012

@author: georgianadinu
'''

import numpy as np
import gzip as gzip
from warnings import warn
from scipy.sparse import csr_matrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.matrix.dense_matrix import DenseMatrix

def get_partitions(sorted_list):
    
    new_elem_index_list = [0]
    for i in range(1, len(sorted_list) - 1):
        if sorted_list[i] != sorted_list[i-1]:
            new_elem_index_list.append(i)
    new_elem_index_list.append(len(sorted_list))
    
    keys = [sorted_list[new_elem_index_list[i]] for i in range(0, len(new_elem_index_list) - 1)]
    
    return keys, new_elem_index_list
            

def list2dict(list_):
    return_dict = {}
    
    for idx, word in enumerate(list_):
        if word in return_dict:
            raise ValueError("duplicate string found in list: %s" % (word)) 
        return_dict[word] = idx

    return return_dict
    
def add_items_to_dict(dict_, list_):
    
    no_els = len(dict_)
    for idx, el in enumerate(list_):
        if el in dict_:
            raise ValueError("Found duplicate keys when appending elements to\
                            dictionary.")
        dict_[el] = no_els + idx
    return dict_    
    
def assert_dict_match_list(dict_, list_):    
    
    match_err = ValueError("expected matching dictionary and list structures.")
    
    if not len(list_) == len(dict_):
        raise match_err
    for (k, v) in dict_.iteritems():
        if not list_[v] == k:
            raise match_err
        
    
def assert_shape_consistent(matrix_, id2row, id2column, row2id, column2id):
    
    no_rows = matrix_.mat.shape[0]
    no_cols = matrix_.mat.shape[1]
    
    has_column_maps = column2id or id2column
     
    if not no_rows == len(id2row) or not no_rows == len(row2id):
        raise ValueError("expected consistent shapes: %d %d %d" 
                         % (no_rows, len(id2row), len(row2id)))
    
    if (has_column_maps and 
        (not no_cols == len(id2column) or not no_cols == len(column2id))):
        raise ValueError("expected consistent shapes: %d %d %d" 
                         % (no_cols, len(id2column), len(column2id))) 


def assert_is_instance(object_, class_):
    if not isinstance(object_, class_):
        raise TypeError("expected %s, received %s" % (class_, type(object_)))

def read_sparse_space_data(matrix_file, row2id, column2id, **kwargs):

    if matrix_file.endswith(".gz"):
        f = gzip.open(matrix_file, "rb")
    else:
        f = open(matrix_file, "rb")    
        
    no_lines = sum(1 for line in f if line.strip() != "")
    f.close()
    
    row = np.zeros(no_lines, dtype = np.int32)
    col = np.zeros(no_lines, dtype = np.int32)
    
    if "dtype" in kwargs:
        element_type = kwargs["dtype"]
    else:
        element_type = np.double
        
    data = np.zeros(no_lines, dtype=element_type)
    
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
                    data[i] = element_type(count)
                    i += 1
            if len(line_elements) > 3:
                warn("Invalid input line:%s. Expected 3 fields, ignoring additional ones!" % line.strip())        
            if len(line_elements) < 3:
                raise ValueError("Invalid row: %s, expected at least %d fields" 
                                 % (line.strip(), 3))
    
    f.close()      
    # eliminate the extra zeros created when word1 or word2 is not row2id or col2id!!    
    data = data[0:i]
    row = row[0:i]
    col = col[0:i]

    m = SparseMatrix(csr_matrix( (data,(row,col)), shape = (len(row2id), len(column2id))))
    if m.mat.nnz != i:
        warn("Found 0-counts or duplicate row,column pairs. (Duplicate entries are summed up.)")
    return m

"""
def read_dense_space_data(self, matrix_file, row2id=None, id2row=None, **kwargs):
    #get number of rows and columns
    with open(matrix_file, "rb") as f:
        first_line = f.next()
        no_cols = len(first_line.strip().split()) - 1
        no_rows = sum(1 for line in f if line.strip() != "") + 1
    
    if not row2id is None:
        no_rows = len(row2id)
    
    if "dtype" in kwargs:
        element_type = kwargs["dtype"]
    else:
        element_type = np.double
        
    m = np.mat(np.zeros(shape=(no_rows, no_cols), dtype = element_type))
    
    if row2id is None:
        row2id = {}
        id2row = []
        i = 0 
        with open(matrix_file, "rb") as f:
            for line in f:
                if not line.strip() == "":
                    els = line.strip().split()
                    word = els[0]
                    row2id[word] = i
                    id2row.append(word)
                    m[i, :] = els[1:]
                    i += 1
    else:
        with open(matrix_file, "rb") as f:
            for line in f:
                if not line.strip() == "":
                    els = line.strip().split()
                    word = els[0]
                    if word in row2id:
                        i = row2id[word]
                        m[i, :] = els[1:]

    return DenseMatrix(m), id2row, row2id
"""

def read_dense_space_data(matrix_file, row2id, **kwargs):
    #get number of rows and columns
    with open(matrix_file, "rb") as f:
        first_line = f.next()
        no_cols = len(first_line.strip().split()) - 1
        if no_cols <= 0:
            raise ValueError("Invalid row: %s, expected at least %d fields" 
                                     % (first_line.strip(), 2))

    no_rows = len(row2id)
    row_string_set = set([])
    
    if "dtype" in kwargs:
        element_type = kwargs["dtype"]
    else:
        element_type = np.double
        
    m = np.mat(np.zeros(shape=(no_rows, no_cols), dtype = element_type))
   
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
                    m[i,:] = elements[1:]
                    row_string_set.add(word)
    
    f.close()
                        
    return DenseMatrix(m)
