function [U, beta_U] = generate_smooth_subspace(n, r, beta)
%% generate a (n x r) orthogonal subspace U
% obeying the subspace smoothness condition with parameter (nearly) beta
% via the following Stiefel manifold optimization probelm
% min_{U' * U = I} 0.5 * (\|D * U\|_F^2 - beta * r)^2
% here, we solve this problem using the trust regions algorithm 
% from the Manopt toolbox (which converges very fast)

% Inpute parameters
% n: Number of rows in matrix U
% r: Number of columns in matrix U
% beta: smoothness parameter
%  ------------------------------------------------------------------------
% written by Hailin Wang, 2024/03/01, wanghailin97@163.com
%% Feasibility check (optional, to verify beta is in valid range)
if beta < 0 || beta >= 4
    error('beta is outside feasible range [0, 4)');
end

D = sparse(diff(eye(n))); % First-order difference matrix, (n-1) x n

% Define Stiefel manifold
manifold = stiefelfactory(n, r);

% Define cost function and gradient
problem.M = manifold;
problem.cost = @(U) 0.5 * (norm(D*U, 'fro')^2 - beta * r)^2;
problem.egrad = @(U) 2 * (norm(D*U, 'fro')^2 - beta * r) * (D' * (D * U)); % Euclidean gradient
% Manopt automatically projects the Euclidean gradient onto the manifold's tangent space

% Check gradient consistency (optional, for debugging)
% checkgradient(problem);

% Set optimization options
options.tolgradnorm = 1e-6; % Gradient norm tolerance
options.maxiter = 1000;     % Maximum number of iterations
options.verbosity = 2;      % Verbosity level

% Initial point: random orthogonal matrix
U0 = manifold.rand();

% Run optimization
[U, info] = trustregions(problem, U0, options);
beta_U = L2Inf_norm(diff(U)')^2;
% Check constraint satisfaction
fprintf('Final constraint value: beta = %f\n', beta_U);
fprintf('Orthogonality check: ||U''*U - I||_F = %f\n', norm(U'*U - eye(r), 'fro'));

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

