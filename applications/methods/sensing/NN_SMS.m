function [ X ] = NN_SMS(y, A, dim, gamma, rho, mu, max_mu, maxIter, tol)
%% Nuclear norm based stable matrix sensing (NN_SMS) via ADMM: 
%  min_{X} 1/2*||y-A(X)||_2^2 + gamma*||X||_*,
%  where A denotes the compressed operator, i.e., a series of sensing matrices
%  parameters: 
%     y      : compressed observation, sized num x 1
%     A      : sensing matrix/operator, sized num x m*n, y = A*vec(M)
%     dim    : data size, [m, n]
%     gamma  : regularization parameter for nuclear norm (default: sigma*sqrt(min(m,n)))
%     rho    : scale lifting parameter in ADMM framework
%     mu     : dual parameter in ADMM framework
%     max_mu : maximum of mu
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters
m = dim(1);
n = dim(2);   
% if nargin < 4; gamma   = sigma * sqrt(log(min(m,n)));           end 
if nargin < 5; rho     = 1.1;                                   end     
if nargin < 6; mu      = 1e-4;                                  end  
if nargin < 7; max_mu  = 1e10;                                  end  
if nargin < 8; maxIter = 500;                                   end          
if nargin < 9; tol     = 1e-6;                                  end      

Aty= reshape(A'*y, [m,n]);

%% variables initialization
X  = zeros(m,n);
Z  = X;
M1 = zeros(m,n);

%% main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;  
    Xk = X;
    %% Update X -- solve linear equation via pcg
    B = Aty + mu*Z - M1;
    X = reshape(My_pcg(X(:), B(:), A, mu), dim);
    
    %% Update Z -- solve NN by singular value thresholding
    [Z, nn_Z] = singval_thre(X + M1/mu, gamma/mu);
    
    %% Stop criterion
    dZ  = X - Z;
    chgX = max(abs(Xk(:)-X(:)));
    chg  = max([chgX max(abs(dZ(:)))]);
    if chg < tol
        break;
    end 
    
    %% Update detail display
    if iter == 1 || mod(iter, 10) == 0
        obj = 0.5*norm(y-A*X(:))^2 + gamma * nn_Z;
        err = norm(dZ(:));
        disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
            ', obj=' num2str(obj) ', err=' num2str(err)]);
    end
    
    %% Update M1 mu
    M1 = M1 + mu*dZ;
    mu = min(rho*mu,max_mu);        
end
obj = 0.5*norm(y-A*X(:))^2 + gamma * nn_Z;
err = norm(dZ(:));
disp(['iter ' num2str(iter) ', mu=' num2str(mu) ', obj=' num2str(obj) ', err=' num2str(err)]);
end

%% needed operations
function x = My_pcg(x, b, A, mu)
% preconditional conjugate gradient method solving:
% (A*A+mu)x = b
x = pcg(@(x) Fun(x), b, 1e-4, 1000, [], [], x);   
    function y = Fun(x)
        y = A'*(A*x) + mu*x;
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
