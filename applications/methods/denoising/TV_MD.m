function [ X ] = TV_MD(M, beta, rho, mu, max_mu, maxIter, tol)
%% Total Variation based matrix denoising (TV_MD) via ADMM: 
%  min_{X} 1/2*||X-M||_F^2 + beta*||Dx(X)||_1 + beta*||Dy(X')||_1,
%  where Dx, Dy are the difference operators along x-axis, y-axis
%  parameters: 
%     M      : noisy matrix data sized m x n
%     beta   : regularization parameter of total variation norm (default: sigma/sqrt(min(m,n)))
%     rho    : update parameter in ADMM
%     mu     : dual variable penalty parameter
%     max_mu : maximum of dual variable penalty parameter
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%% parameters
[m, n]  = size(M);
% if nargin < 2; beta    = sigma / sqrt(min(m,n));          end          
if nargin < 3; rho     = 1.1;                             end     
if nargin < 4; mu      = 1e-4;                            end  
if nargin < 5; max_mu  = 1e10;                            end  
if nargin < 6; maxIter = 500;                             end          
if nargin < 7; tol     = 1e-6;                            end         

%% variables initialization
X   = M;
G1  = Diff(X,1);
G2  = Diff(X,2);
M1  = zeros(m,n);
M2  = zeros(m,n);

%% FFT setting
Eny_x = ( abs(psf2otf([+1; -1], [m,n])) ).^2; 
Eny_y = ( abs(psf2otf([+1, -1], [m,n])) ).^2;
T = Eny_x + Eny_y; 

%% main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;  
    Xk = X;
    %% Update X -- solve difference operation involved linear equation by FFT 
    B = M + DiffT(mu*G1-M1,1) + DiffT(mu*G2-M2,2);
    X = real( ifftn( fftn(B)./(1 + mu*T) ) );
  
    %% Updata G1 G2 -- soft-thresholding operator
    G1 = soft_thre(Diff(X,1) + M1/mu, beta/mu);
    G2 = soft_thre(Diff(X,2) + M2/mu, beta/mu);
    
    %% Stop criterion
    dG1 = Diff(X,1) - G1;
    dG2 = Diff(X,2) - G2;
    chgX = max(abs(Xk(:)-X(:)));
    chg  = max([chgX max(abs(dG1(:))) max(abs(dG2(:)))]);
    if chg < tol
        break;
    end 
    
    %% Update detail display
    if iter == 1 || mod(iter, 10) == 0
        obj = 0.5*norm(M(:)-X(:))^2 + beta * (norm(G1(:),1) + norm(G2(:),1));
        err = norm(dG1(:)) + norm(dG2(:));
        disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
            ', obj=' num2str(obj) ', err=' num2str(err)]);
    end
    
    %% Update M1 M2 mu
    M1 = M1 + mu*dG1;
    M2 = M2 + mu*dG2;
    mu = min(rho*mu,max_mu);    
    
end
obj = 0.5*norm(M(:)-X(:))^2 + beta * (norm(G1(:),1) + norm(G2(:),1));
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

function DX = Diff(X, direction)
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

function DtX = DiffT(X, direction)
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

