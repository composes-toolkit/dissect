'''
Created on Oct 5, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
import unittest
import numpy as np
from composes.semantic_space.space import Space
from composes.matrix.sparse_matrix import SparseMatrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.transformation.external import Matlab
import numpy
from scipy.sparse.csr import csr_matrix
from composes.semantic_space.operation import ScalingOperation
from composes.transformation.external.matlab_nmf import MatlabNmf


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        try:
            self.matlab_command = "matlab"
            self.matlab_args = [ '-nodisplay', '-nosplash','-nodesktop', '-r']
            import subprocess
            subprocess.call([self.matlab_command] + self.matlab_args + ['quit'])
        except OSError:
            pass

#    def test_matlab_dense(self):
#        x = np.array([[1,2,3],[4,0,5]])
#        space = Space(DenseMatrix(x), ["a", "b"], ["f1","f2", "f3"])
#        expected = np.array([[2,4,6],[8,0,10]])
#        try:
#            tr = Matlab(
#'''function y = double(x)
#    y = 2*x; 
#end''',     
#            self.matlab_command, self.matlab_args)
#            tr.create_operation = lambda : ScalingOperation(tr)
#            tr.uses_column_stats = False
#            obtained = space.apply(tr)
#            numpy.testing.assert_array_equal(expected, obtained.cooccurrence_matrix.mat)
#            
#        except OSError, e:
#            if e.errno == 2:
#                #Matlab not found
#                pass
#            else:
#                raise
            
            
#    def test_matlab_sparse(self):
#        x = csr_matrix((2,3))
#        x[0,0]=1
#        x[0,1]=2
#        x[0,2]=3
#        x[1,0]=4
#        x[1,1]=0
#        x[1,2]=5
#        space = Space(SparseMatrix(x), ["a", "b"], ["f1","f2", "f3"])
#        #it returns a dense matrix!
#        expected = np.array([[2,4,6],[8,0,10]])
#        try:
#            tr = Matlab(
#'''function y = double(x)
#    y = 2*x; 
#end''', 
#            self.matlab_command, self.matlab_args)
#            tr.create_operation = lambda : ScalingOperation(tr)
#            tr.uses_column_stats = False
#            obtained = space.apply(tr)
#            numpy.testing.assert_array_equal(expected, obtained.cooccurrence_matrix.mat)
#            
#        except OSError, e:
#            if e.errno == 2:
#                #Matlab not found
#                pass
#            else:
#                raise
#        
    def test_nmf(self):
        test_cases = [np.mat([[1,2,3],[2,4,6],[4,17,13]], dtype = np.double)]#,
#                      np.mat([[1,0,0]], dtype = np.double)]
        
        try:
            for in_mat in test_cases:
                red = MatlabNmf(2, False, self.matlab_command, self.matlab_args)
                red_op = red.create_operation()
                d_mat = DenseMatrix(in_mat)
                s_mat = SparseMatrix(in_mat)
                
                wd_mat=red_op.apply(d_mat)
                hd_mat = red_op.project(DenseMatrix(np.array
                                                ([[1,0,0],[0,1,0],[0,0,1]])))
                red_op = red.create_operation()
                ws_mat=red_op.apply(s_mat)
                hs_mat = red_op.project(DenseMatrix(np.array
                                                ([[1,0,0],[0,1,0],[0,0,1]])))
                
    
                print wd_mat.mat
                print hd_mat.mat
                print ws_mat.mat
                print hs_mat.mat 
                np.testing.assert_array_almost_equal(wd_mat.mat,
                                                     ws_mat.mat, 2)
                np.testing.assert_array_almost_equal(hd_mat.mat,
                                                     hs_mat.mat, 2)
        except OSError, e:
            if e.errno == 2:
                #Matlab not found
                pass
            else:
                raise
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_top_feat_selection']
    unittest.main()
