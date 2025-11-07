function [ M_est, X, Y ] = sLRMF_SMC(M, Mask, rank, lambda, lr, maxIter, tol)
%% Smoothed low-rank matrix factorization based stable maatrix completion (sLRMF_SMC) via gradient descent: 
%  min_{X,Y} 1/2p*||(X*Y')(Mask)-M(Mask)||_F^2 + lambda/2p*||D(X)||_F^2 + lambda/2p*||D(Y)||_F^2 + 1/8*||X'X-Y'*Y||_F^2,
%  where D is the difference operator
%  parameters: 
%     M      : observed noisy incompleted matrix data sized m x m
%     Mask   : the mask index matrix 
%     rank   : number of rank 
%     sigma  : noise deviation
%     lambda : regularization parameter (default: sigma*sqrt(min(m,n))*p)
%     lr     : learning rate (step size)
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters   
[m, n] = size(M);
p = sum(Mask(:))/(m*n);
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
	
	%% updates
    Grad_Xk = ((q*Xk*Yk').*Mask - qM_Mask) * Yk + q * lambda * DiffT(Diff(Xk)) + 0.5 * Xk * (Xk'*Xk-Yk'*Yk);
    X = Xk - lr*Grad_Xk;
    Grad_Yk = ((q*Xk*Yk').*Mask - qM_Mask)' * Xk + q * lambda * DiffT(Diff(Yk)) + 0.5 * Yk * (Yk'*Yk-Xk'*Xk);
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
