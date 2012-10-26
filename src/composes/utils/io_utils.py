'''
Created on Oct 16, 2012

@author: nghia
'''

import numpy as np
import pickle
import os
import gzip as gzip
from warnings import warn
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.utils.space_utils import assert_is_instance

def save(object_, file_name):
    create_parent_directories(file_name)
    with open(file_name,'w') as f:
        pickle.dump(object_, f, 2)

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
    parts = file_name.split("/")
    if (len(parts) > 1):
        parent_dir = "/".join(parts[0:-1])
        if (not os.path.exists(parent_dir)):
            print parent_dir
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


def read_tuple_list(data_file, **kwargs):
    field_list = []
    result = []
    if "fields" in kwargs:
        field_list = kwargs["fields"]
    
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if (line != ""):
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
                
def print_list(list_, file_name):
    with open(file_name,'w') as f:
        for item in list_:
            f.write(item + "\n")
            
            
def print_cooc_mat_sparse_format(matrix_, id2row, id2column, file_prefix):
    matrix_file = "%s.%s" %(file_prefix, "sm")
    
    with open(matrix_file, 'w') as f: 
        if isinstance(matrix_, SparseMatrix):
            if not id2column:
                raise ValueError("Cannot print matrix with no column info in sparse format!")
            mat = matrix_.mat

            data = mat.data
            row_indices = mat.indptr
            col_indices = mat.indices
            
            row_index = 0
            next_row = row_indices[1]
            row = id2row[0]  
            for i in xrange(len(data)):
                while i == next_row:
                    row_index +=1
                    next_row = row_indices[row_index + 1]
                    row = id2row[row_index]  
                col = id2column[col_indices[i]]
                f.write("%s\t%s\t%f\n" %(row,col,data[i]))
        else:
            raise NotImplementedError("Not implemented\n")        

def print_cooc_mat_dense_format(matrix_, id2row, file_prefix):
    matrix_file = "%s.%s" %(file_prefix, "dm")
    
    with open(matrix_file, 'w') as f: 
        for i, row in enumerate(id2row):
            v = DenseMatrix(matrix_[i]).mat.flat
            line = "\t".join([row] + [repr(v[j]) for j in range(len(v))])
            f.write("%s\n" % (line))

