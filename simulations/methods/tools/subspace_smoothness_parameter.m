function beta = subspace_smoothness_parameter(M, r)
%% compute the subspace smoothness condition parameter 'beta' of a rank-r matrix 
% definition: for M = U_{m,r} * S_{r,r} * V_{n,r}', the
% subspace smoothness condition is defined as follows:
% max_{i=1,...,r}||DU_{.,j}||_F <= sqrt(beta)
% and max_{i=1,...,r}||DV_{.,j}||_F <= sqrt(beta)
%  where 0 <= beta < 4
% the minium value of the beta that satisfies the above inequality
[U,~,V] = svds(M, r);
beta = max(L2Inf_norm(diff(U)')^2, L2Inf_norm(diff(V)')^2);
end

function a = L2Inf_norm(X)
% compute the l_{2,inf} norm of X
% ||X||_{2,inf} := max_{i} ||X(i,:)|| = max_{i} ||X'*e_i||
a = 0;
for i = 1:size(X,1)
   temp = norm(X(i,:));
   a = max(a,temp);
end
end