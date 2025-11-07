function [ X ] = TV_SMC(M, Mask, beta, rho, mu, max_mu, maxIter, tol)
%% Nuclear norm based stable matrix completion (NN_SMC) via ADMM: 
%  min_{X} 1/2p*||X(Mask)-M(Mask)||_F^2 +  beta*||Dx(X)||_1 + beta*||Dy(X)||_1,
%  where Dx, Dy are the difference operators along x-axis, y-axis
%  parameters: 
%     M      : observed noisy incompleted matrix data sized m x n
%     Mask   : the mask index matrix 
%     beta   : regularization parameter for total variation norm (default: sigma/sqrt(min(m,n))*p)
%     rho    : scale lifting parameter in ADMM framework
%     mu     : dual parameter in ADMM framework
%     max_mu : maximum of mu
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters
[m, n] = size(M);
p = sum(Mask(:))/(m*n);
% if nargin < 3; beta    = sigma / sqrt(min(m,n)*p);          end  
if nargin < 4; rho     = 1.2;                               end     
if nargin < 5; mu      = 1e-4;                              end  
if nargin < 6; max_mu  = 1e10;                              end  
if nargin < 7; maxIter = 500;                               end          
if nargin < 8; tol     = 1e-6;                              end      

Mask_c = ones(m,n) - Mask;
q = 1/p;
qM_Mask = q * (M .* Mask);

%% variables initialization
X  = zeros(m,n);
Z  = X;
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
	T = qM_Mask + (mu*Z-M3);
	X = (p/(1+p*mu)) * (T.*Mask) + (1/mu) * (T.*Mask_c);
	
    %% Update Z -- solve difference equation via FFT
    B = enDiffT(mu*G1 - M1,1) +  enDiffT(mu*G2 - M2,2) + (mu*X + M3);
    Z = real(ifftn( fftn(B)./(mu*(1+Eny)) ));
    
    %% Update G1,G2 -- solve L1 by soft thresholding
    G1 = soft_thre(enDiff(Z,1) + M1/mu, beta/mu);
    G2 = soft_thre(enDiff(Z,2) + M2/mu, beta/mu);
	
    %% Stop criterion
    dG1 = enDiff(Z,1) - G1;
    dG2 = enDiff(Z,2) - G2;
	dZ  = X - Z;
    chgX = max(abs(Xk(:)-X(:)));
    chg = max([ chgX max(abs(dG1(:))) max(abs(dG2(:))) ]);
    if chg < tol
        break;
    end 
    
    %% Update detail display
    if iter == 1 || mod(iter, 10) == 0
        obj = 0.5*norm((X(:)-M(:)).*Mask(:))^2 + beta * (norm(G1(:),1) + norm(G2(:),1));
        err = norm(dG1(:)) + norm(dG2(:));
        disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
            ', obj=' num2str(obj) ', err=' num2str(err)]);
    end
    
    %% Update M1 M2 M3 mu
    M1 = M1 + mu*dG1;
    M2 = M2 + mu*dG2;
    M3 = M3 + mu*dZ;
    mu = min(rho*mu,max_mu);        
end
obj = 0.5*norm((X(:)-M(:)).*Mask(:))^2 + beta * (norm(G1(:),1) + norm(G2(:),1));
err = norm(dG1(:)) + norm(dG2(:));
disp(['iter ' num2str(iter) ', mu=' num2str(mu) ', obj=' num2str(obj) ', err=' num2str(err)]);
end

%% needed operations
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
