function [M_est, X, Y] = LRMF_SMS(y, A, dim, rank, lambda, lr, maxIter, tol)
%% Low-rank matrix factorization based stable matrix sensing (LRMF_SMS) via gradient descent: 
%  min_{X,Y} 1/2*||y-A(X*Y')||_2^2 + lambda/2*||X||_F^2 + lambda/2*||Y||_F^2 + 1/8*||X'X-Y'*Y||_F^2,
%  parameters: 
%     y      : compressed observation, sized num x 1
%     A      : sensing matrix/operator, sized num x m*n, y = A*vec(M)
%     dim    : data size, [m, n]
%     rank   : number of rank 
%     sigma  : noise deviation
%     lambda : regularization parameter (default: sigma*sqrt(min(m,n)))
%     lr     : learning rate (step size)
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
% written by Hailin Wang, 2024/03/01, wanghailin97@163.com
%% parameters
m = dim(1);
n = dim(2); 
% if nargin < 5; lambda  = sigma * sqrt(min(m,n));            end 
if nargin < 6; lr      = 0.001;                             end           
if nargin < 7; maxIter = 1000;                              end           
if nargin < 8; tol     = 1e-6;                              end           

Aty = reshape(A'*y, [m, n]);

%% spectral initialization
[U0,S0,V0] = svds(Aty, rank);
X = U0*sqrt(S0);
Y = V0*sqrt(S0);

%% main loops
iter = 0;
while iter < maxIter
    iter = iter + 1;
    Xk = X; Yk = Y; 
    
    Mk = Xk*Yk';
    Rk = reshape(A'*(A*Mk(:)), [m,n]) - Aty; % residual term
    Bk = Xk' * Xk - Yk' * Yk; % balance term
    
    %% gradient descent
    Grad_Xk = Rk * Yk  + lambda * Xk + 0.5 * Xk * Bk;
    X = Xk - lr*Grad_Xk;
    Grad_Yk = Rk' * Xk + lambda * Yk - 0.5 * Yk * Bk;
    Y = Yk - lr*Grad_Yk;
    
    %% stop criterion  
    chgX = max(abs(Xk(:) - X(:)));
    chgY = max(abs(Yk(:) - Y(:)));
    if iter == 1 || mod(iter, 100) == 0
        disp(['iter ' num2str(iter) ',||Xk-X||=' num2str(chgX,'%2.3e') ',||Yk-Y||=' num2str(chgY,'%2.3e')]);
    end
    if chgX<tol && chgY<tol
        break;
    end
end
disp(['iter ' num2str(iter) ',||Xk-X||=' num2str(chgX,'%2.3e') ',||Yk-Y||=' num2str(chgY,'%2.3e')]);

M_est = X*Y';
end
