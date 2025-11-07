function [ Smooth_X, X,S ] = PRPCA(Y, lambda, tau, P, Q, opts)
% Solving the following PRPCA problem via Proximal Gradient Decent Algorithm
%   min_{X,S} 0.5*\|Y - P*X*Q'- S\|_F^2 + lambda*\|X\|_* + tau*\|S\|_1
% where P, Q are the interpolation matrices
%  parameters: 
%     Y      : noisy outlier-mixed matrix data sized m x n 
%     lambda : regularization parameter of nuclear norm (default: sigma*sqrt(min(m,n)))
%     tau    : regularization parameter of sparse (oulier) matrix (default: sigma*sqrt(log(min(m,n))))
%     P      : row projection matrix (default: interpolation matrix)
%     Q      : column projection matrix (default: interpolation matrix)
%     opts   : the parameters in PGD framework
%  ------------------------------------------------------------------------
%% default paremeters setting 
[M, N] = size(Y);
% if nargin < 2; lambda  = sigma * sqrt(min(m,n));            end    
% if nargin < 3; tau     = sigma * sqrt(log(min(m,n)));       end   
if nargin < 4; P  = Normalized_Interpolation_Matrix(M);   end 
if nargin < 5; Q  = Normalized_Interpolation_Matrix(N);   end   

t         = 1;   % step size (associated with Lipschitz constant)
beta      = 0.5; % used to update L for finding proper step size
tol       = 1e-5; 
max_iter  = 1000;
detail    = 1;
if ~exist('opts', 'var')
    opts = [];
end   
if isfield(opts, 't');        t        = opts.t;        end
if isfield(opts, 'beta');     beta     = opts.beta;     end
if isfield(opts, 'tol');      tol      = opts.tol;      end
if isfield(opts, 'max_iter'); max_iter = opts.max_iter; end
if isfield(opts, 'detail');   detail   = opts.detail;   end

%% variables initialization
m = size(P,2); n = size(Q,2);
X = zeros(m, n); S = zeros(M, N);
F = 0.5 * norm(Y - P*X*Q'-S, 'fro')^2;

%% main loop
iter = 0;
while iter < max_iter
    iter = iter + 1;  
    Xk = X; Sk = S;Fk = F;
    %% proximal gradient decent
    while 1
        Grad_S = P*Xk*Q' + Sk - Y;
        Grad_X = P'*Grad_S*Q;
        [X, nn_X] = singval_thre(Xk-t*Grad_X, lambda*t);
        S = soft_thre(Sk-t*Grad_S, tau*t);
        F = 0.5 * norm(Y - P*X*Q'-S, 'fro')^2;
        if F <= Fk + sum(sum(Grad_X.*(X - Xk))) + sum(sum(Grad_S.*(S - Sk))) ...
                   + 1/(2*t) * (norm(X - Xk, 'fro')^2 + norm(S - Sk, 'fro')^2)
            break
        end
        t = beta * t;
    end
    
    %% stop criterion
    chgX = max(abs(Xk(:)-X(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg  = max([chgX chgS]);
    if chg < tol
        break;
    end 
    
    %% update detail display
    if detail
        if iter == 1 || mod(iter, 10) == 0
            obj = lambda * nn_X + tau * norm(S,1);
            disp(['iter = ' num2str(iter) ', obj = ' num2str(obj)...
                ', chgX = ' num2str(chgX) ', chgS = ' num2str(chgS)]); 
        end
    end
    
end

if detail
    obj = lambda * nn_X + tau * norm(S,1);
    disp(['iter = ' num2str(iter) ', obj = ' num2str(obj)...
        ', chgX = ' num2str(chgX) ', chgS = ' num2str(chgS)]); 
end

Smooth_X = P * X * Q';

end


function M = Normalized_Interpolation_Matrix(n)
% interpolation matrix generate function
p = ceil(n/2);
M = zeros(n,p);
% i=1
M(1,1) = 1; M(2,1) = 1; M(3,1) = 0.5;
% i=2:n-1
for i = 2:p-1
    M(2*i-1,i) = 0.5; M(2*i,i) = 1; M(2*i+1,i) = 0.5;
end
% i=n
if mod(n,2) ==0
    M(n-1,p) = 0.5; M(n,p) = 1;
else
    M(n,p) = 0.5;
end
end

function [ X, nn_X ] = singval_thre(Y, tau)
% singular value thresholding operator
% min_X tau*||X||_* + 0.5*||X-Y||_F^2
% 
[U,S,V] = svd(Y,'econ');
S = diag(S);
svp = length(find(S>tau));
if svp>=1
    S = S(1:svp) - tau;
    X = U(:,1:svp)*diag(S)*V(:,1:svp)';
    nn_X = sum(S);
else
    X = zeros(size(Y));
    nn_X = 0;
end
end

function x = soft_thre(b,lambda)
% soft-thresholding operator
% min_x lambda*||x||_1+0.5*||x-b||_2^2
%
x = max(0,b-lambda)+min(0,b+lambda);
end


