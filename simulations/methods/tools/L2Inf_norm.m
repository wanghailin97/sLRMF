function a = L2Inf_norm(X)
% compute the l_{2,inf} norm of X
% ||X||_{2,inf} := max_{i} ||X(i,:)|| = max_{i} ||X'*e_i||
a = 0;
for i = 1:size(X,1)
   temp = norm(X(i,:));
   a = max(a,temp);
end
end