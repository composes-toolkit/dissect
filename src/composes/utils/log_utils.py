'''
Created on Oct 15, 2012

@author: georgianadinu
'''

def print_matrix_info(logger_, matrix_, ident_level):
    ident = " " * ident_level
    logger_.info("\n%sMatrix type:%s" % (ident, type(matrix_)))
    logger_.info("\n%sMatrix shape:%sx%s" % (ident, matrix_.shape[0], 
                                          matrix_.shape[1]))
    
    