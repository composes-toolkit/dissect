'''
Created on Sep 18, 2012

@author: georgianadinu
'''

from numbers import Number
from numbers import Integral
import numpy as np

def is_numeric(operand):
    return isinstance(operand, (Number, np.number))

def is_integer(operand):
    return isinstance(operand, Integral)
