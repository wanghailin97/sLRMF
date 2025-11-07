function x = trimming(b, lambda)
% trimming operator
% 
x = b.*(b>-lambda).*(b<lambda) + lambda*(b>=lambda) - lambda*(b<=-lambda);
end