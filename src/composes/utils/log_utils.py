'''
Created on Oct 15, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

import logging

from numpy import double

from composes.utils.io_utils import create_parent_directories


def config_logging(file_name, level=logging.INFO, format_=""):
    if not file_name is None:
        create_parent_directories(file_name)
        logging.basicConfig(filename=file_name, level=level, format=format_)
        logging.debug("start logging")


def get_ident(delim, ident_level):
    return delim * ident_level


def print_matrix_info(logger_, matrix_, ident_level, intro_string):
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident + intro_string
    ident = ident + delim

    logger_string += (", %sMatrix type:%s" % (ident, type(matrix_).__name__))
    logger_string += (", %sMatrix shape:%sx%s" % (ident, matrix_.shape[0],
                                                  matrix_.shape[1]))

    if type(matrix_).__name__ == "SparseMatrix":
        perc_nnz = 100 * matrix_.mat.nnz / double(matrix_.shape[0] * matrix_.shape[1])
        logger_string += (", %sPerc. non-zero entries:%d" % (ident, perc_nnz))

    logger_.debug(logger_string)


def get_learner_info(learner):
    logger_string = ""

    if hasattr(learner, '_intercept'):
        logger_string += ("Using intercept:%s" % (learner._intercept))

    if hasattr(learner, '_crossvalidation'):
        logger_string += ("Using crossvalidation:%s" % (learner._crossvalidation))

        if learner._crossvalidation and hasattr(learner, '_folds'):
            logger_string += ("Using number of folds:%s" % (learner._folds))

    return logger_string


def print_composition_model_info(logger_, model, intro_string, *args):
    name = type(model).__name__
    learner = ''
    if hasattr(model, '_regression_learner'):
        learner = "Using regression:%s, %s, " % (type(model.regression_learner).__name__,
                                                 get_learner_info(model.regression_learner))

    logger_.info('%s %s %s' % (intro_string, name, learner))


def print_info(logger_, ident_level, text):
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident + ""

    logger_string += "\n%s%s" % (ident, text)
    logger_.info(logger_string)


def print_transformation_info(logger_, trans, ident_level, intro_string):
    delim = "  "
    ident = get_ident(delim, ident_level)
    logger_string = ident + intro_string
    ident = ident + delim

    logger_string += (", %sTransformation type:%s" % (ident, type(trans).__name__))

    if hasattr(trans, '_reduced_dimension'):
        logger_string += (", %sReduced dimension:%s" % (ident, trans.reduced_dimension))

    logger_.info(logger_string)


def print_time_info(logger_, end, beg, *args):
    logger_.info("Timing: %s seconds" % (end - beg))
     
