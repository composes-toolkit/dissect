'''
Created on Sep 21, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

"""
Wrappers around psutil functions that display memory usage information. 
"""
import numpy as np
from os import getpid
import psutil

def get_mem_usage():
    p = psutil.Process(getpid())
    return p.get_memory_info()[0]/np.double(1024*1024)    