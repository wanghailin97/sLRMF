function [ M_est, S, X, Y ] = LRMF_SRPCA_v1(M, rank, lambda, tau, zeta, lr, maxIter, tol)
%% Low-rank matrix factorization based stable RPCA (LRMF_SRPCA) via gradient descent: 
%  min_{X,Y} 1/2*||X*Y'+S-M||_F^2 + tau*||S||_1 lambda/2*||X||_F^2 + lambda/2*||Y||_F^2 + 1/8*||X'X-Y'*Y||_F^2,
%  parameters: 
%     M      : observed sparse(outlier)-mixed noisy data sized m x n
%     rank   : number of rank 
%     lambda : regularization parameter (default: sigma*sqrt(min(m,n)))
%     tau    : regularization parameter of sparse (oulier) matrix (default: sigma*sqrt(log(min(m,n))))
%     lr     : learning rate (step size)
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
% written by Hailin Wang, 2024/03/01, wanghailin97@163.com
%% parameters
[m, n] = size(M);
dim = [m, n];
% if nargin < 3; lambda  = sigma * sqrt(min(m,n));            end    
% if nargin < 4; tau     = sigma * sqrt(log(min(m,n)));       end   
% if nargin < 5; zeta    = 1 + sigma * sqrt(log(min(m,n)));   end  
if nargin < 6; lr      = 0.001;                               end           
if nargin < 7; maxIter = 1000;                                end           
if nargin < 8; tol     = 1e-6;                                end           

%% spectral initialization
Tim_M = trimming(M, zeta);
[U0,S0,V0] = svds(Tim_M, rank);
X = U0*sqrt(S0);
Y = V0*sqrt(S0);
S = soft_thre(M - X*Y', tau);

%% main loops
iter = 0;
while iter < maxIter
    iter = iter + 1;
    Xk = X; Yk = Y; 
    
    Rk = Xk * Yk' + S - M; % residual term
    Bk = Xk' * Xk - Yk' * Yk; % balance term
	
	%% gradient descent
    Grad_Xk = Rk * Yk  + lambda * Xk + 0.5 * Xk * Bk;
    X = Xk - lr*Grad_Xk;
    Grad_Yk = Rk' * Xk + lambda * Yk - 0.5 * Yk * Bk;
    Y = Yk - lr*Grad_Yk;
    
    %% soft thresholding 
    S = soft_thre(M - X*Y', tau);
    
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

%% needed operations
function x = soft_thre(b,lambda)
% soft-thresholding operator
% min_x lambda*||x||_1+0.5*||x-b||_2^2
%
x = max(0,b-lambda)+min(0,b+lambda);
end
