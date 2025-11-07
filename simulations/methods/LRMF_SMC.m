function [ M_est, X, Y ] = LRMF_SMC(M, Mask, rank, lambda, lr, maxIter, tol)
%% Low-rank matrix factorization based stable matrix completion (LRMF_SMC) via gradient descent: 
%  min_{X,Y} 1/2p*||(X*Y')(Mask)-M(Mask)||_F^2 + lambda/2p*(||X||_F^2 + ||Y||_F^2) + 1/8*||X'X-Y'*Y||_F^2,
%  parameters: 
%     M      : observed noisy incompleted matrix data sized m x n
%     Mask   : the mask index matrix 
%     rank   : number of rank 
%     lambda : regularization parameter (default: sigma*sqrt(min(m,n)/p))
%     lr     : learning rate (step size)
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
% written by Hailin Wang, 2024/03/01, wanghailin97@163.com
%% parameters  
[m, n] = size(M);
dim = [m, n];
p = sum(Mask(:))/(prod(dim));
% if nargin < 4; lambda  = sigma * sqrt(min(m,n)*p);          end  
if nargin < 5; lr      = 0.001;                             end           
if nargin < 6; maxIter = 1000;                              end           
if nargin < 7; tol     = 1e-6;                              end           

q = 1/p;
qM_Mask = q * (M .* Mask);

%% spectral initialization
[U0,S0,V0] = svds(qM_Mask, rank);
X = U0*sqrt(S0);
Y = V0*sqrt(S0);

%% main loops
iter = 0;
while iter < maxIter
    iter = iter + 1;
    Xk = X; Yk = Y; 
    
    Rk =q * (Xk * Yk') .* Mask - qM_Mask; % residual term
    Bk = Xk' * Xk - Yk' * Yk; % banlance term
	
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
