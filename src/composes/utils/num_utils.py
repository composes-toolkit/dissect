'''
Created on Sep 18, 2012

@author: georgianadinu
'''

from numbers import Number
import numpy as np

def is_numeric(operand):
    return isinstance(operand, (Number, np.number))

