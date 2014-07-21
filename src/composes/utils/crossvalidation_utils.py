'''
Created on Oct 9, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

from random import shuffle

def get_split_indices(range_len, fold):

    if fold <= 0:
        raise ValueError("Strictly positive number of folds required, received %s:"
                         % fold)

    indices_list = []
    if range_len < fold:
        return get_split_indices(range_len, range_len)

    range_ = range(range_len)
    shuffle(list(range_))
    current_index = 0
    for i in range(fold):
        if i < len(range_)%fold:
            slice_length = range_len // fold + 1
        else:
            slice_length = range_len // fold

        indices_list.append(range_[current_index:current_index + slice_length])
        current_index += slice_length

    return indices_list

def get_submatrix_list(matrix_, indices_list):
    return [matrix_[indices, :] for indices in indices_list]

