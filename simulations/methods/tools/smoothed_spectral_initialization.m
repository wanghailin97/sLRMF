function [X, Y] = smoothed_spectral_initialization(M, r, gamma)
% smoothed spectral initialization via a proximal projection with smoothness regularization
[m, n] = size(M);

[U, S, V] = svds(M, r);
X = U * sqrt(S);
Y = V * sqrt(S);

D1 = diff(eye(m));
D2 = diff(eye(n));
[V1,s1] = eig(D1'*D1, 'vector');
[V2,s2] = eig(D2'*D2, 'vector');
X = V1 * ( (V1' * X) ./ (repmat(gamma*s1, 1, r) + 1));
Y = V2 * ( (V2' * Y) ./ (repmat(gamma*s2, 1, r) + 1));
end