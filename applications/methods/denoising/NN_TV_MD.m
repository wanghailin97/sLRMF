function [ X ] = NN_TV_MD(M, gamma, beta, rho, mu, max_mu, maxIter, tol)
%% Nuclear norm plus total variation based matrix denoising (NN_TV_MD) via ADMM: 
%  min_{X} 1/2*||X-M||_F^2 + gamma*||X||_* + beta*||Dx(X)||_1 + beta*||Dy(X)||_1,
%  where Dx, Dy are the difference operators along x-axis, y-axis
%  parameters: 
%     M      : noisy matrix data sized m x n 
%     gamma  : regularization parameter of nuclear norm (default: sigma*sqrt(min(m,n)))
%     beta   : regularization parameter of total variation norm (default: sigma/sqrt(min(m,n)))
%     rho    : update parameter in ADMM
%     mu     : dual variable penalty parameter
%     max_mu : maximum of dual variable penalty parameter
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters
[m, n]  = size(M);
% if nargin < 2; gamma   = sigma * sqrt(min(m,n));          end    
% if nargin < 3; beta    = sigma / sqrt(min(m,n));          end 
if nargin < 4; rho     = 1.1;                             end     
if nargin < 5; mu      = 1e-4;                            end  
if nargin < 6; max_mu  = 1e10;                            end  
if nargin < 7; maxIter = 500;                             end          
if nargin < 8; tol     = 1e-6;                            end         

%% variables initialization
X   = M;
G1  = enDiff(X,1);
G2  = enDiff(X,2);
Z   = X;
M1  = zeros(m,n);
M2  = zeros(m,n);
M3  = zeros(m,n);

%% FFT setting
Eny_x = ( abs(psf2otf([+1; -1], [m,n])) ).^2; 
Eny_y = ( abs(psf2otf([+1, -1], [m,n])) ).^2;
Eny = Eny_x + Eny_y; 

%% main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;  
    Xk = X;
    %% Update X -- solve difference operation involved linear equation by FFT 
    B = M + enDiffT(mu*G1-M1,1) + enDiffT(mu*G2-M2,2) + (mu*Z - M3);
    X = real( ifftn( fftn(B)./(1 + mu + mu*Eny) ) );
    
    %% Update Z -- solve NN by singular value thresholding
    [Z, nn_Z] = singval_thre(X + M3/mu, gamma/mu);
    
    %% Updata G1 G2 -- soft-thresholding operator
    G1 = soft_thre(enDiff(X,1) + M1/mu, beta/mu);
    G2 = soft_thre(enDiff(X,2) + M2/mu, beta/mu);
    
    %% Stop criterion
    dG1 = enDiff(X,1) - G1;
    dG2 = enDiff(X,2) - G2;
    dZ  = X - Z;
    chgX = max(abs(Xk(:)-X(:)));
    chg  = max([ chgX max(abs(dG1(:))) max(abs(dG2(:))) max(abs(dZ(:))) ]);
    if chg < tol
        break;
    end 
    
    %% Update detail display
    if iter == 1 || mod(iter, 10) == 0
        obj = 0.5*norm(M(:)-X(:))^2 + gamma * nn_Z + beta * (norm(G1(:),1) + norm(G2(:),1));
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
obj = 0.5*norm(M(:)-X(:))^2 + gamma * nn_Z + beta * (norm(G1(:),1) + norm(G2(:),1));
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

function x = soft_thre(y, tau)
% soft-thresholding operator
% min_x tau*||x||_1 + 0.5*||x-b||_2^2
%
x = max(0,y-tau)+min(0,y+tau);
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

