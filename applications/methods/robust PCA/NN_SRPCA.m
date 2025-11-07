function [ X, S ] = NN_SRPCA(M, gamma, tau, maxIter, tol)
%% Nuclear norm based stable RPCA (NN_SRPCA) via AltMin: 
%  min_{X} 1/2*||X+S-M||_F^2 + tau*||S||_1 + gamma*||X||_*,
%  parameters: 
%     M      : observed sparse(outlier)-mixed noisy data sized m x n
%     gamma  : regularization parameter of nuclear norm (default: sigma*sqrt(min(m,n)))
%     tau    : regularization parameter of sparse (oulier) matrix (default: sigma*sqrt(log(min(m,n))))
%     maxIter: maximum number of iterations
%     tol    : tolerance/convergence factor
%  ------------------------------------------------------------------------
%% parameters
[m, n] = size(M);
% if nargin < 2; gamma  = sigma * sqrt(min(m,n));             end    
% if nargin < 3; tau     = sigma * sqrt(log(min(m,n)));       end     
if nargin < 4; maxIter = 500;                               end          
if nargin < 5; tol     = 1e-6;                              end      

%% variables initialization
X = zeros(m,n);
S = zeros(m,n);

%% main loop
iter = 0;
while iter < maxIter
    iter = iter + 1;  
    Xk = X; Sk = S;
    
    %% Update X -- solve NN by singular value thresholding
    [X, nn_X] = singval_thre(M-S, gamma);
    
    %% Update S -- solve L1 by soft thresholding
    S = soft_thre(M-X, tau);
    
    %% Stop criterion
    chgX = max(abs(Xk(:)-X(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg  = max([chgX chgS]);
    if chg < tol
        break;
    end 
    
    %% Update detail display
    if iter == 1 || mod(iter, 10) == 0
        obj = 0.5*norm(X+S-M,'fro')^2 + gamma * nn_X + tau * norm(S,1);
        disp(['iter ' num2str(iter) ', obj=' num2str(obj) ', err=' num2str(chg)]);
    end       
end
obj = 0.5*norm(X+S-M,'fro')^2 + gamma * nn_X + tau * norm(S,1);
disp(['iter ' num2str(iter) ', obj=' num2str(obj) ', err=' num2str(chg)]);
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
