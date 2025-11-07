function res=mtimes(A,x)
if ~isa(A,'IdentityWHT_partitioned')
    error('In A*x, A should be a object of name PermuteWHT_partitioned');
end

if ~A.adjoint
    
    x      = reshape(x, A.m, A.L);
    X      = [x; zeros(A.M - A.m, A.L)];
    FX     = fdWHtrans(IdentityPerm(X,A.perm));
%     FX=fwht(X(A.perm,:));
    res     = FX(A.picks);
else
    
    FY = zeros(A.M, A.L);
    FY(A.picks) = x;
    res  = IdentityPerm(fdWHtrans(FY),A.perm);
%     res(A.perm,:)=fwht(FY);
    res = res(1:A.m,:);
    res = res(:);
end
end

function outMatrix = IdentityPerm(inMatrix,list)
[m,L]=size(inMatrix);
outMatrix = zeros([m,L]);
for i=1:length(list)
    outMatrix(list(i),:) = -1*inMatrix(list(i),:);
end
end