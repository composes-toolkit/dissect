'''
Created on Sep 26, 2012

@author: georgianadinu
'''

def list2dict(list_):
    return_dict = {}
    
    for idx, word in enumerate(list_):
        if word in return_dict:
            raise ValueError("duplicate string found in list: %s" % (word)) 
        return_dict[word] = idx

    return return_dict
    
def assert_dict_match_list(dict_, list_):    
    
    match_err = ValueError("expected matching dictionary and list structures.")
    
    if not len(list_) == len(dict_):
        raise match_err
    for (k, v) in dict_.iteritems():
        if not list_[v] == k:
            raise match_err
        
    
def assert_shape_consistent(self, matrix_, id2row, id2column,
                            row2id, column2id):
    
    no_rows = matrix_.mat.shape[0]
    no_cols = matrix_.mat.shape[1]
    
    has_column_maps = column2id or id2column
     
    if not no_rows == len(id2row) or not no_rows == len(row2id):
        raise ValueError("expected consistent shapes: %d %d %d" 
                         % (no_rows, len(id2row), len(row2id)))
    
    if (has_column_maps and 
        (not no_rows == len(id2row) or not no_rows == len(row2id))):
        raise ValueError("expected consistent shapes: %d %d %d" 
                         % (no_cols, len(id2column), len(column2id))) 

    
def assert_is_instance(object_, class_):
    if not isinstance(object_, class_):
        raise TypeError("expected %s, received %s" % (class_, type(object_)))    