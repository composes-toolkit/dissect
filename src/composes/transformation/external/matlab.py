import re
from scipy.io import savemat, loadmat
from os import path,getcwd,chdir
from subprocess import call
from composes.matrix.dense_matrix import DenseMatrix
from composes.transformation.external.external import External
from composes.utils import io_utils


class Matlab(External):
    '''
    Performs a transformation of the Matrix using matlab code
    '''
    def __init__(self, code, matlab_exec="matlab", matlab_args=[ '-nodisplay',
                                                    '-nosplash',
                                                    '-nodesktop',
                                                    '-r'], tmp_path="/tmp", ascii=False):
        '''
        code: A matlab function that takes a matrix and returns the transformed
        matrix
        matlab_exec: shell command to run matlab
        tmp_path: a path to temporarily store the matrices
        '''
        self.code = code
        self.matlab_exec = matlab_exec
        m = re.finditer('function \[?((?:\w|\d|_|,|\s)+)\]?\s*=\s*([\w\d_]+)', code)\
            .next()
        self.ret_vals =  [x.strip() for x in re.split("\s*,\s*",m.group(1))]
        self.function_name = m.group(2)
        self.tmp_path = tmp_path
        self.matlab_args = matlab_args
        self.ascii = ascii
    
    def create_operation(self):
        raise NotImplementedError()
    
    def apply(self, matrix):
        import tempfile
        tmp_script = tempfile.NamedTemporaryFile(dir=self.tmp_path, 
                                                      suffix='.m')
        tmp_script_name = path.basename(tmp_script.name)[:-2]
        tmp_script_path = tmp_script.name
        if self.ascii:
            tmp_mat_file = tempfile.NamedTemporaryFile(dir=self.tmp_path, 
                                                        suffix='.sm')
            tmp_mat_path = tmp_mat_file.name
            #Export matrix to tmp_path in ascii format
            io_utils.print_cooc_mat_sparse_format(matrix, 
                {i: str(i+1) for i in range(matrix.get_shape()[0])}, 
                {i: str(i+1) for i in range(matrix.get_shape()[1])},
                tmp_mat_path[:-3])
            command = """
            function {script_name}
            %load matrix
            load '{mat_path}';
            x = spconvert({mat_name});
            """
        else:
            tmp_mat_file = tempfile.NamedTemporaryFile(dir=self.tmp_path, 
                                                        suffix='.mat')
            tmp_mat_path = tmp_mat_file.name
            #Export matrix to tmp_path
            savemat(tmp_mat_path, {'x': matrix.mat})
            #Run matlab code on exported matrix
            command = """
            function {script_name}
            %load matrix
            load '{mat_path}'
            """
            
        command += \
        """
        %call the user function
        [{ret_vals}]={func_name}(x);
        %save results and quit
        save '{mat_path}' {ret_vals};
        quit; 
        end
        %define user function
        {code}
        """
        command = command.format(mat_path=tmp_mat_path,
        mat_name=path.basename(tmp_mat_path).split(".")[0],
                   code=self.code, ret_vals=" ".join(self.ret_vals), 
                   func_name=self.function_name, script_name=tmp_script_name)
        with open(tmp_script_path, 'w') as f:
            f.write(command)
        
        cwd = getcwd()
        chdir(self.tmp_path)
        call([self.matlab_exec]+ self.matlab_args +
             [tmp_script_name])
        chdir(cwd)
        #Load back the result       
        ret = loadmat(tmp_mat_path)
        #FIXME: create a MatrixFactory that build the correct wrapper
        rvs=map(DenseMatrix, map(ret.__getitem__, self.ret_vals))
        if len(rvs)==1:
            return rvs[0]
        else:
            return rvs
