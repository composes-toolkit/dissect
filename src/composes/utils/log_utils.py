'''
Created on Oct 15, 2012

@author: georgianadinu
'''

from numpy import double


def get_ident(delim, ident_level):
    return delim * ident_level

def print_matrix_info(logger_, matrix_, ident_level, intro_string):
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident + intro_string
    ident = ident + delim
    
    logger_string += ("\n%sMatrix type:%s" % (ident, type(matrix_).__name__))
    logger_string += ("\n%sMatrix shape:%sx%s" % (ident, matrix_.shape[0], 
                                          matrix_.shape[1]))
    
    if type(matrix_).__name__ == "SparseMatrix":
        perc_nnz = 100 * matrix_.mat.nnz/double(matrix_.shape[0]*matrix_.shape[1])
        logger_string += ("\n%sPerc. non-zero entries:%d" % (ident, perc_nnz))    
    
    logger_.info(logger_string)

        
def get_learner_info(learner, ident):
    logger_string = ""
    
    if hasattr(learner, '_intercept'):
        logger_string += ("\n%sUsing intercept:%s" % (ident, learner._intercept))

    if hasattr(learner, '_crossvalidation'):
        logger_string += ("\n%sUsing crossvalidation:%s" % (ident, learner._crossvalidation))
        
        if learner._crossvalidation and hasattr(learner, '_folds'):
            logger_string += ("\n%sUsing number of folds:%s" % (ident, learner._folds))

    return logger_string    
            
def print_composition_model_info(logger_, model, ident_level, intro_string):
 
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident + intro_string
    ident = ident + delim
    
    logger_.info(logger_string)
                
    print_name(logger_, model, ident_level, "Composition model type:")
    
    logger_string = ""
    if hasattr(model, '_regression_learner'):
        logger_string += ("\n%sUsing regression:%s" % (ident, 
                                                       type(model.regression_learner).__name__))
        logger_string += get_learner_info(model.regression_learner, ident + delim)
        
    logger_.info(logger_string)
             
def print_transformation_info(logger_, trans, ident_level, intro_string):
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident + intro_string
    ident = ident + delim
    
    logger_string += ("\n%sTransformation type:%s" % (ident, type(trans).__name__))
    
    if hasattr(trans, '_reduced_dimension'):
        logger_string += ("\n%sReduced dimension:%s" % (ident, trans.reduced_dimension))
        
    
    logger_.info(logger_string)
    
def print_info(logger_, ident_level, text):
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident + ""

    logger_string += "\n%s%s" % (ident, text)
    logger_.info(logger_string)    

def print_name(logger_, object_, ident_level, intro_string):
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident + intro_string
    ident = ident + delim
        
    logger_string += ("\n%s%s" % (ident, type(object_).__name__))

    logger_.info(logger_string)     
 
def print_time_info(logger_, end, beg, ident_level):
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident
    logger_string += ("\n%sTiming:%s seconds" % (ident, end - beg))
    
    logger_.info(logger_string)
     
