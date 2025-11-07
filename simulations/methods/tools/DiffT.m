function DtX = DiffT(X) 
% the transpose of the first order difference operation
DtX = [-X(1,:);-diff(X);X(end,:)];
end