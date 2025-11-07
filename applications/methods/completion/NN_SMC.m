function [ X ] = NN_SMC(M, Mask, gamma, rho, mu, max_mu, maxIter, tol)
%% Nuclear norm based stable matrix completion (NN_SMC) via ADMM: 
%  min_{X} 1/2p*||X(Mask)-M(Mask)||_F^2 + gamma*||X||_*,
%  parameters: 
%     M      : observed noisy incompleted matrix data sized m x n
%     Mask   : the mask index matrix 
%     gamma  : regularization parameter for nuclear norm (default: sigma*sqrt(min(m,n)/p))
%     rho    : scale lifting parameter in ADMM framework
%     mu     : dual parameter in ADMM framework
%     max_mu : maximum of mu
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters
[m, n] = size(M);
p = sum(Mask(:))/(m*n);
% if nargin < 3; gamma  = sigma * sqrt(min(m,n)/p);          end  
if nargin < 4; rho     = 1.1;                              end     
if nargin < 5; mu      = 1e-4;                             end  
if nargin < 6; max_mu  = 1e10;                             end  
if nargin < 7; maxIter = 500;                              end          
if nargin < 8; tol     = 1e-6;                             end    

Mask_c = ones(m,n) - Mask;
q = 1/p;
qM_Mask = q * (M .* Mask);

%% variables initialization
X = zeros(m,n);
Z = X;
M1 = zeros(m,n);

%% main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;  
    Xk = X;
    %% Update X 
    T = qM_Mask + (mu*Z - M1);
    X = (p/(1+p*mu)) * (T.*Mask) + (1/mu) * (T.*Mask_c);
    
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
        obj = 0.5*norm((X(:)-M(:)).*Mask(:))^2 + gamma * nn_Z;
        err = norm(dZ(:));
        disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
            ', obj=' num2str(obj) ', err=' num2str(err)]);
    end
    
    %% Update M1 mu
    M1 = M1 + mu*dZ;
    mu = min(rho*mu,max_mu);        
end
obj = 0.5*norm((X(:)-M(:)).*Mask(:))^2 + gamma * nn_Z;
err = norm(dZ(:));
disp(['iter ' num2str(iter) ', mu=' num2str(mu) ', obj=' num2str(obj) ', err=' num2str(err)]);
end

%% needed operations
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
