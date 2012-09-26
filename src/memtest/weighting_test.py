'''
Created on Sep 20, 2012

@author: georgianadinu
'''
import numpy as np
import time
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.weighting.plmi import PpmiWeighting
from composes.utils.mem_utils import get_mem_usage

"""
 RUN /usr/bin/time -v python weighting_test.py
"""
 
def test_plog():
    x = np.random.rand(10000,10000)
    x = DenseMatrix(x)
    print "Done creating"
    print "Using %s MB " % (get_mem_usage())
    
    x = PpmiWeighting().apply(x)
    print "Done weighting"
    print "Using %s MB " % (get_mem_usage())    
    
    print x.mat.shape 
    
if __name__ == '__main__':
    test_plog()
    