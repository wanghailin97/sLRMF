function [ X ] = TV_SMS(y, A, dim, beta, rho, mu, max_mu, maxIter, tol)
%% Nuclear norm based stable matrix sensing (NN_SMS) via ADMM: 
%  min_{X} 1/2*||y-A(X)||_2^2 +  beta*||Dx(X)||_1 + beta*||Dy(X)||_1,
%  where A denotes the compressed operator, i.e., a series of sensing matrices
%  and Dx, Dy are the difference operators along x-axis, y-axis
%  parameters: 
%     y      : compressed observation, sized num x 1
%     A      : sensing matrix/operator, sized num x m*n, y = A*vec(M)
%     dim    : data size, [m, n]
%     beta   : regularization parameter for total variation norm (default: sigma/sqrt(min(m,n)))
%     rho    : scale lifting parameter in ADMM framework
%     mu     : dual parameter in ADMM framework
%     max_mu : maximum of mu
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters 
m = dim(1);
n = dim(2);    
% if nargin < 4; beta    = sigma / sqrt(min(m,n));          end 
if nargin < 5; rho     = 1.1;                       end     
if nargin < 6; mu      = 1e-4;                      end  
if nargin < 7; max_mu  = 1e10;                      end  
if nargin < 8; maxIter = 500;                       end          
if nargin < 9; tol     = 1e-6;                      end      

Aty= reshape(A'*y, [m,n]);

%% variables initialization
X  = zeros(m,n);
G1 = enDiff(X,1);
G2 = enDiff(X,2);
M1 = zeros(m,n);
M2 = zeros(m,n);

%% main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;  
    Xk = X;
    %% Update X -- solve quadtric equation via pcg
    B = Aty + enDiffT(mu*G1 - M1,1) +  enDiffT(mu*G2 - M2,2);
    X = reshape(My_pcg(X(:), B(:), A, mu, dim), dim);
    
    %% Update G1,G2,G3 -- solve L1 by soft thresholding
    G1 = soft_thre(enDiff(X,1) + M1/mu, beta/mu);
    G2 = soft_thre(enDiff(X,2) + M2/mu, beta/mu);
    
    %% Stop criterion
    dG1 = enDiff(X,1) - G1;
    dG2 = enDiff(X,2) - G2;
    chgX = max(abs(Xk(:)-X(:)));
    chg = max([ chgX max(abs(dG1(:))) max(abs(dG2(:))) ]);
    if chg < tol
        break;
    end 
    
    %% Update detail display
    if iter == 1 || mod(iter, 10) == 0
        obj = 0.5*norm(y-A*X(:))^2 + beta * (norm(G1(:),1) + norm(G2(:),1));
        err = norm(dG1(:)) + norm(dG2(:));
        disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
            ', obj=' num2str(obj) ', err=' num2str(err)]);
    end
    
    %% Update M1 M2 M3 mu
    M1 = M1 + mu*dG1;
    M2 = M2 + mu*dG2;
    mu = min(rho*mu,max_mu);        
end
obj = 0.5*norm(y-A*X(:))^2 + beta * (norm(G1(:),1) + norm(G2(:),1));
err = norm(dG1(:)) + norm(dG2(:));
disp(['iter ' num2str(iter) ', mu=' num2str(mu) ', obj=' num2str(obj) ', err=' num2str(err)]);
end

%% needed operations
function x = My_pcg(x, b, A, mu, dim)
% preconditional conjugate gradient method solving:
% (A*A+mu*D1'D1+mu*D2'D2+mu*D3'*D3)x = b
x = pcg(@(x) Fun(x), b, 1e-4, 1000, [], [], x);   
    function y = Fun(x)
        X = reshape(x,dim);
        df = enDiffT(enDiff(X,1),1) +  enDiffT(enDiff(X,2),2);
        y = A'*(A*x) + mu*df(:);
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
