function [ X ] = NN_MD(M, gamma)
%% Nuclear norm based matrix denoising (NN_MD): 
%  min_{X} 1/2*||X-M||_F^2 + gamma*||X||_*,
%  which has closed-form solution via the singular value thresholding
%  parameters: 
%     M      : noisy matrix data sized m x n 
%     gamma  : regularization parameter of nuclear norm (default: sigma*sqrt(min(m,n)))
%  ------------------------------------------------------------------------
%% parameters 
[m, n] = size(M);   
% if nargin < 2; gamma  = sigma * sqrt(min(m,n));            end 

%% main
[U,S,V] = svd(M,'econ');
S = diag(S);
svp = length(find(S>gamma));
if svp>=1
    S = S(1:svp)-gamma;
    X = U(:,1:svp)*diag(S)*V(:,1:svp)';
else
    X = zeros(size(B));
end
end