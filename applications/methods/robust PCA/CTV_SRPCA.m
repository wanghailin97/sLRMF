function [ X ] = CTV_SRPCA(M, lambda, tau, rho, mu, max_mu, maxIter, tol)
%% Correlated total variation based stable RPCA (CTV_SRPCA) via ADMM: 
%  min_{X} 1/2*||X+S-M||_F^2 + tau*||S||_1 + lambda*||Dx(X)||_* + lambda*||Dx(X)||_*,
%  where Dx, Dy are the difference operators along x-axis, y-axis
%  parameters: 
%     M      : noisy outlier-mixed matrix data sized m x n 
%     lambda : regularization parameter of correlated total variation norm (default: sigma*sqrt(min(m,n)))
%     tau    : regularization parameter of sparse (oulier) matrix (default: sigma*sqrt(log(min(m,n))))
%     rho    : update parameter in ADMM
%     mu     : dual variable penalty parameter
%     max_mu : maximum of dual variable penalty parameter
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters
[m, n] = size(M);
% if nargin < 2; lambda  = sigma * sqrt(min(m,n));            end    
% if nargin < 3; tau     = sigma * sqrt(log(min(m,n)));       end   
if nargin < 4; rho     = 1.1;                               end     
if nargin < 5; mu      = 1e-4;                              end  
if nargin < 6; max_mu  = 1e10;                              end  
if nargin < 7; maxIter = 500;                               end          
if nargin < 8; tol     = 1e-6;                              end      

%% variables initialization
X  = zeros(m,n);
S  = zeros(m,n);
G1 = enDiff(X,1);
G2 = enDiff(X,2);
M1 = zeros(m,n);
M2 = zeros(m,n);

%% FFT setting
Eny_x = ( abs(psf2otf([+1; -1], [m,n])) ).^2; 
Eny_y = ( abs(psf2otf([+1, -1], [m,n])) ).^2;
Eny = Eny_x + Eny_y; 

%% main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;  
    Xk = X; Sk = S;
	%% Updata X -- solve difference equation via FFT
    B = (M-S) + enDiffT(mu*G1 - M1, 1) +  enDiffT(mu*G2 - M2, 2);
    X = real(ifftn( fftn(B)./(1+mu*Eny) ));
    
    %% Update S -- solve L1 by soft thresholding
    S = soft_thre(M-X, tau);
	
    %% Update G1,G2 -- solve L1 by soft thresholding
    [G1, nn_G1] = singval_thre(enDiff(X,1) + M1/mu, lambda/mu);
    [G2, nn_G2] = singval_thre(enDiff(X,2) + M2/mu, lambda/mu);
	
    %% Stop criterion
    dG1 = enDiff(X,1) - G1;
    dG2 = enDiff(X,2) - G2;
    chgX = max(abs(Xk(:)-X(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([ chgX chgS max(abs(dG1(:))) max(abs(dG2(:))) ]);
    if chg < tol
        break;
    end 
    
    %% Update detail display
    if iter == 1 || mod(iter, 10) == 0
        obj = 0.5*norm(X+S-M,'fro')^2 + tau*norm(S,1) + lambda * (nn_G1+nn_G2);
        err = norm(dG1(:)) + norm(dG2(:));
        disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
            ', obj=' num2str(obj) ', err=' num2str(err)]);
    end
    
    %% Update M1 M2 mu
    M1 = M1 + mu*dG1;
    M2 = M2 + mu*dG2;
    mu = min(rho*mu,max_mu);        
end
obj = 0.5*norm(X+S-M,'fro')^2 + tau*norm(S,1) + lambda * (nn_G1+nn_G2);
err = norm(dG1(:)) + norm(dG2(:));
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

function x = soft_thre(b,lambda)
% soft-thresholding operator
% min_x lambda*||x||_1+0.5*||x-b||_2^2
%
x = max(0,b-lambda)+min(0,b+lambda);
end

function DX = enDiff(X, direction)
% enclosed difference operator
dim = size(X);
index_first = repmat({':'},1,ndims(X));
index_first(direction) = {1};
index_end = repmat({':'},1,ndims(X));
index_end(direction) = {dim(direction)};

slice = X(index_first{:}) - X(index_end{:});
DX  = diff(X,1,direction);
DX  = cat(direction,DX,slice);
end

function DtX = enDiffT(X, direction)
% transpose of the enclosed difference operator
dim = size(X);
index_first = repmat({':'},1,ndims(X));
index_first(direction) = {1};
index_end = repmat({':'},1,ndims(X));
index_end(direction) = {dim(direction)};

slice = X(index_first{:}) - X(index_end{:});
DX  = diff(X,1,direction);
DtX  = cat(direction,-slice,-DX);
end
