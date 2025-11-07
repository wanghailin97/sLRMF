%% demo test 
% for proposed smoothed low-rank matrix factorization (sLRMF) 
% v.s. baseline low-rank matrix factorization (LRMF)

clear;close all;
addpath(genpath(pwd))

%% synthesis simultaneous low-rank and smooth matrix
n = 100; r = 10;
init_beta = 0.01;
U0 = generate_smooth_subspace(n, r, init_beta);
V0 = generate_smooth_subspace(n, r, init_beta);
[~,S0,~] = svds(randn(n,n), r); 
M0 = U0 * S0 * V0';
beta = subspace_smoothness_parameter(M0, r);

%% noisy observation
sigma = 0.5; % can be quite large
noisy_M = M0 + sigma * randn(n, n);
rmse_obs = norm(noisy_M - M0, 'fro') / n;

%% run low-rank matrix factorization
disp('===== run LRMF =====');
est_LRMF = LRMF(noisy_M, r, sigma * sqrt(n));
rmse_LRMF = norm(est_LRMF - M0, 'fro') / n;

%% run smoothed low-rank matrix factorization
disp('===== run sLRMF =====');
est_sLRMF = sLRMF(noisy_M, r, 5 * sigma * sqrt(n));
rmse_sLRMF = norm(est_sLRMF - M0, 'fro') / n;

%% results display
fprintf('\n');
fprintf('=============== Results ===============\n');
fprintf('  %-8s  %-8s   %-8s   %-8s   \n', 'Method', 'Obs.', 'LRMF', 'sLRMF');
fprintf('  %-6s   %8.4f  %8.4f  %8.4f  \n', 'RMSE', rmse_obs, rmse_LRMF, rmse_sLRMF);
fprintf('=============== Results ===============\n');

figure(1)
subplot 221; imshow(M0); colormap summer; title('original');
subplot 222; imshow(noisy_M); colormap summer; title('noisy');
subplot 223; imshow(est_LRMF); colormap summer; title(['LRMF (RMSE=' num2str(rmse_LRMF,3) ')']);
subplot 224; imshow(est_sLRMF); colormap summer; title(['sLRMF (RMSE=' num2str(rmse_sLRMF,3) ')']);
