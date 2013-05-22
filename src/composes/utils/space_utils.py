'''
Created on Sep 26, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
            

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



