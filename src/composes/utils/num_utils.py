'''
Created on Sep 18, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

from numbers import Number
import numpy as np
import six

def is_numeric(operand):
    return isinstance(operand, (Number, np.number))

def is_integer(operand):
    return isinstance(operand, six.integer_types) or isinstance(operand, np.integer)

