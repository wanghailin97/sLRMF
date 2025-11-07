function [ M_est, X, Y ] = sLRMF_SMS(y, A, dim, rank, lambda, lr, maxIter, tol)
%% Smoothed low-rank matrix factorization based stable matrix sensing (sLRMF_SMS) via gradient descent: 
%  min_{X,Y} 1/2*||A(X*Y')-y||_2^2 + lambda/2*||D(X)||_F^2 + lambda/2*||D(Y)||_F^2 + 1/8*||X'X-Y'*Y||_F^2,
%  where A denotes the compressed operator, i.e., a series of sensing matrices,
%  and D is the difference operator
%  parameters: 
%     y      : compressed observation, sized m x 1
%     A      : sensing matrix/operator, sized num x m*n, y = A*vec(M)
%     dim    : data size, [m, n]
%     rank   : number of rank 
%     lambda : regularization parameter (default: sigma*sqrt(min(m,n)))
%     lr     : learning rate (step size)
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters     
m = dim(1);
n = dim(2); 
% if nargin < 5; lambda  = sigma * sqrt(min(m,n));            end 
if nargin < 6; lr      = 0.001;                             end           
if nargin < 7; maxIter = 1000;                              end           
if nargin < 8; tol     = 1e-6;                              end            

Aty = reshape(A'*y, [m,n]);

%% spectral initialization
[U0,S0,V0] = svds(Aty, rank);
X = U0*sqrt(S0);
Y = V0*sqrt(S0);

%% main loops
iter = 0;
while iter < maxIter
    iter = iter + 1;
    Xk = X; Yk = Y; 
    XkYkt = Xk*Yk';
    Grad_Xk = (reshape(A'*(A*XkYkt(:)), [m,n]) - Aty) * Yk + lambda * DiffT(Diff(Xk)) + 0.5 * Xk * (Xk'*Xk-Yk'*Yk);
    X = Xk - lr*Grad_Xk;
    Grad_Yk = (reshape(A'*(A*XkYkt(:)), [m,n]) - Aty)' * Xk + lambda * DiffT(Diff(Yk)) + 0.5 * Yk * (Yk'*Yk-Xk'*Xk);
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

%% needed operations
function DX = Diff(X) 
% first order difference operation, default: the row difference
DX = diff(X);
end

function DtX = DiffT(X) 
% the transpose of the first order difference operation
DtX = [-X(1,:);-diff(X);X(end,:)];
end
