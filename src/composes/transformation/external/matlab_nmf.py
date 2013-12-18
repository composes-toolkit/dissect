from composes.transformation.external.matlab import Matlab
from composes.transformation.dim_reduction.dimensionality_reduction import DimensionalityReduction

class MatlabNmf(DimensionalityReduction):
    name = "nmf"
    
    def __init__(self, reduced_dimension, normd=False, 
                 matlab_exec="matlab", matlab_args=[ '-nodisplay',
                                                    '-nosplash',
                                                    '-nodesktop',
                                                    '-r'], 
                 tmp_path="/home/german.kruszewski/tmp",
                 ascii=False):
        super(MatlabNmf, self).__init__(reduced_dimension)
        self.normd = normd
        self.matlab_args = matlab_args
        self.tmp_path = tmp_path
        self.matlab_exec = matlab_exec
        self.ascii = ascii
        
        
    def apply(self, matrix):
        code = '''
        function [W, H_pinv] = do_run_nmf(M)
            [W,H_pinv] = run_nmf(M, {0}, {1});
        end'''.format(self.reduced_dimension, 'true' if self.normd else 'false')
        code += run_nmf
        code += nmf_m
        op = Matlab(code, self.matlab_exec, self.matlab_args, self.tmp_path, 
                    self.ascii)
        return op.apply(matrix)
    
run_nmf = \
r'''
function [W, H_pinv] = run_nmf(M, rank, normd)

    s = size(M);
    V=M;
    %V=full(M);
    %V = zeros(max(M(:,1)), max(M(:,2)) ) ;

    %for i=1:s(1),
    %    V(M(i,1),M(i,2)) = M(i,3) ;
    %end
    %clear M ;

    if normd,
        V = V./sum(sum(V)) ;
    end

    Hinit = V(randperm(rank),:) ;
    Winit = V(:, randperm(rank)) ;
    size(Winit) ;
    size(Hinit) ;

    [W,H] = nmf(V, Winit, Hinit, 0.0000001,36000,15) ;

    if normd,
        descr = strcat('norm.rank.', num2str(rank)) ;
    else
        descr = strcat('rank.', num2str(rank)) ;
    end
    W=full(W);
    H_pinv = pinv(full(H)) ;
end
'''

nmf_m = \
r'''function [W,H] = nmf(V,Winit,Hinit,tol,timelimit,maxiter)

% NMF by alternative non-negative least squares using projected gradients
% Author: Chih-Jen Lin, National Taiwan University

% W,H: output solution
% Winit,Hinit: initial solution
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

W = Winit; H = Hinit; initt = cputime;

gradW = W*(H*H') - V*H'; gradH = (W'*W)*H - W'*V;
initgrad = norm([gradW; gradH'],'fro');
fprintf('Init gradient norm %f\n', initgrad); 
tolW = max(0.001,tol)*initgrad; tolH = tolW;

for iter=1:maxiter,
  % stopping condition
  projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
  if projnorm < tol*initgrad | cputime-initt > timelimit,
    break;
  end
  
  [W,gradW,iterW] = nlssubprob(V',H',W',tolW,1000); W = W'; gradW = gradW';
  if iterW==1,
    tolW = 0.1 * tolW;
  end

  [H,gradH,iterH] = nlssubprob(V,W,H,tolH,1000);
  if iterH==1,
    tolH = 0.1 * tolH; 
  end

  if rem(iter,10)==0, fprintf('.'); end
end
fprintf('\nIter = %d Final proj-grad norm %f\n', iter, projnorm);
end

function [H,grad,iter] = nlssubprob(V,W,Hinit,tol,maxiter)

% H, grad: output solution and gradient
% iter: #iterations used
% V, W: constant matrices
% Hinit: initial solution
% tol: stopping tolerance
% maxiter: limit of iterations

H = Hinit; WtV = W'*V; WtW = W'*W; 

alpha = 1; beta = 0.1;
for iter=1:maxiter,  
  grad = WtW*H - WtV;
  disp('gradient norm:')
  disp(normest(grad));
  projgrad = norm(grad(grad < 0 | H >0));
  if projgrad < tol,
    break
  end

  % search step size 
  for inner_iter=1:20,
    Hn = max(H - alpha*grad, 0); d = Hn-H;
    gradd=sum(sum(grad.*d)); dQd = sum(sum((WtW*d).*d));
    suff_decr = 0.99*gradd + 0.5*dQd < 0;
    if inner_iter==1,
      decr_alpha = ~suff_decr; Hp = H;
    end
    if decr_alpha, 
      if suff_decr,
    H = Hn; break;
      else
    alpha = alpha * beta;
      end
    else
      if ~suff_decr | Hp == Hn,
    H = Hp; break;
      else
    alpha = alpha/beta; Hp = Hn;
      end
    end
  end
end

if iter==maxiter,
  fprintf('Max iter in nlssubprob\n');
end
end
'''

