function [ X ] = CTV_SMC(M, Mask, lambda, rho, mu, max_mu, maxIter, tol)
%% Correlated total variation based stable matrix completion (CTV_SMC) via ADMM: 
%  min_{X} 1/2p*||X(Mask)-M(Mask)||_F^2 + lambda*||Dx(X)||_* + lambda*||Dy(X)||_*,
%  where Dx, Dy are the difference operators along x-axis, y-axis
%  parameters: 
%     M      : observed noisy incompleted matrix data sized m x n
%     Mask   : the mask index matrix 
%     lambda : regularization parameter for correlated total variation norm  (default: sigma*sqrt(min(m,n))*p)
%     rho    : scale lifting parameter in ADMM framework
%     mu     : dual parameter in ADMM framework
%     max_mu : maximum of mu
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters
[m, n] = size(M);
p = sum(Mask(:))/(m*n);
% if nargin < 3; lambda  = sigma * sqrt(min(m,n)/p);          end  
if nargin < 4; rho     = 1.1;                               end     
if nargin < 5; mu      = 1e-4;                              end  
if nargin < 6; max_mu  = 1e10;                              end  
if nargin < 7; maxIter = 500;                               end          
if nargin < 8; tol     = 1e-6;                              end      

Mask_c = ones(m,n) - Mask;
q = 1/p;
qM_Mask = q * (M .* Mask);

%% variables initialization
X  = zeros(m,n);
Z = X;
G1 = enDiff(X,1);
G2 = enDiff(X,2);
M1 = zeros(m,n);
M2 = zeros(m,n);
M3 = zeros(m,n);

%% FFT setting
Eny_x = ( abs(psf2otf([+1; -1], [m,n])) ).^2; 
Eny_y = ( abs(psf2otf([+1, -1], [m,n])) ).^2;
Eny = Eny_x + Eny_y; 

%% main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;  
    Xk = X;
	%% Updata X
	T = qM_Mask + (mu*Z - M3);
	X = (p/(1+p*mu)) * (T.*Mask) + (1/mu) * (T.*Mask_c);
	
    %% Update Z -- solve difference equation via FFT
    B = enDiffT(mu*G1 - M1, 1) +  enDiffT(mu*G2 - M2, 2) + (mu*X + M3);
    Z = real(ifftn( fftn(B)./(mu*(1+Eny)) ));
	
    %% Update G1,G2 -- solve L1 by soft thresholding
    [G1, nn_G1]= singval_thre(enDiff(Z,1) + M1/mu, lambda/mu);
    [G2, nn_G2] = singval_thre(enDiff(Z,2) + M2/mu, lambda/mu);
	
    %% Stop criterion
    dG1 = enDiff(Z,1) - G1;
    dG2 = enDiff(Z,2) - G2;
	dZ = X - Z;
    chgX = max(abs(Xk(:)-X(:)));
    chg = max([ chgX max(abs(dG1(:))) max(abs(dG2(:))) max(abs(dZ(:))) ]);
    if chg < tol
        break;
    end 
    
    %% Update detail display
    if iter == 1 || mod(iter, 10) == 0
        obj = 0.5*norm((X(:)-M(:)).*Mask(:))^2 + lambda * (nn_G1+nn_G2);
        err = norm(dG1(:)) + norm(dG2(:)) + norm(dZ(:));
        disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
            ', obj=' num2str(obj) ', err=' num2str(err)]);
    end
    
    %% Update M1 M2 M3 mu
    M1 = M1 + mu*dG1;
    M2 = M2 + mu*dG2;
    M3 = M3 + mu*dZ;
    mu = min(rho*mu,max_mu);        
end
obj = 0.5*norm((X(:)-M(:)).*Mask(:))^2 + lambda * (nn_G1+nn_G2);
err = norm(dG1(:)) + norm(dG2(:)) + norm(dZ(:));
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
