%%%%% test the low-rank and/or smooth based matrix estimation methods for image outlier removaling task %%%%%
% including: 
% 1) nuclear norm (NN)
% 2) low-rank matrix factorization (LRMF)
% 3) total variation (TV)
% 4) joint nuclear norm and total variation (NN+TV)
% 5) correlated total variation (CTV)
% 6) projected RPCA (PRPCA)
% 7) *** the proposed smoothed low-rank matrix factorization (sLRMF) ***

close all; clear;clc;
addpath(genpath('./lib'));
addpath(genpath('./methods/robust PCA'));

% test images from the BSD300 dataset
imgDir = './BSD300-images';
D= dir(fullfile(imgDir,'*.jpg'));

id_of_img = 3;
img_name = fullfile(imgDir,D(id_of_img).name);
img = double(imread(img_name))/255;
[m, n] = size(img); 

% experiments settings
outlier_ratio = 0.1;
sigma = 0.05;
outlier_mixed_img = imnoise(img,'salt & pepper',outlier_ratio);
noisy_outlier_mixed_img = outlier_mixed_img + sigma * randn(m,n);

% Omega = randperm(m*n, round(outlier_ratio*m*n));
% s_min = 0.8; s_max = 1.6;
% O = sign(rand(m,n)-0.5) .* (s_min + (s_max-s_min) * rand(m, n));
% Outlier = zeros(m, n);
% Outlier(Omega) = O(Omega);
% noisy_outlier_mixed_img = img + Outlier + sigma * randn(m,n);

% evaluate the results of each method by RMSE and PSNR
RMSE = zeros(1,8);
PSNR = zeros(1,8);

%% run all methods 
% 1. nuclear norm (NN)
gamma = sigma * sqrt(min(m,n));
tau = 0.5*sigma * sqrt(log(min(m,n)));
disp('----- run NN -----');
img_est_1 = NN_SRPCA(noisy_outlier_mixed_img, gamma, tau);
RMSE(1) = norm(img_est_1 - img, 'fro') / sqrt(m * n);
PSNR(1) = 10 * log10( 1 / RMSE(1)^2 );

% 2. low-rank matrix factorization (LRMF)
rank = 150;
lambda = sigma * sqrt(min(m,n));
tau = 0.5*sigma * sqrt(log(min(m,n)));
lr = 0.001;
maxIter = 2000;
tol = 1e-5;
disp('----- run LRMF -----');
img_est_2 = LRMF_SRPCA(noisy_outlier_mixed_img, rank, lambda, tau, lr, maxIter,tol);
RMSE(2) = norm(img_est_2 - img, 'fro') / sqrt(m * n);
PSNR(2) = 10 * log10( 1 / RMSE(2)^2 );

% 3. total variation (TV)
beta = sigma / sqrt(min(m,n));
tau = 0.5*sigma * sqrt(log(min(m,n)));
disp('----- run TV -----');
img_est_3 = TV_SRPCA(noisy_outlier_mixed_img, beta, tau);
RMSE(3) = norm(img_est_3 - img, 'fro') / sqrt(m * n);
PSNR(3) = 10 * log10( 1 / RMSE(3)^2 );

% 4. joint nuclear norm and total variation (NN+TV)
gamma = sigma * sqrt(min(m,n));
beta = sigma / sqrt(min(m,n));
tau = 0.5*sigma * sqrt(log(min(m,n)));
disp('----- run NN+TV -----');
img_est_4 = NN_TV_SRPCA(noisy_outlier_mixed_img, gamma, beta, tau);
RMSE(4) = norm(img_est_4 - img, 'fro') / sqrt(m * n);
PSNR(4) = 10 * log10( 1 / RMSE(4)^2 );

% 5. correlated total variation (CTV)
lambda = sigma * sqrt(min(m,n));
tau = 0.5*sigma * sqrt(log(min(m,n)));
disp('----- run CTV -----');
img_est_5 = CTV_SRPCA(noisy_outlier_mixed_img, lambda, tau);
RMSE(5) = norm(img_est_5 - img, 'fro') / sqrt(m * n);
PSNR(5) = 10 * log10( 1 / RMSE(5)^2 );

% 6. projected RPCA (PRPCA)
lambda = sigma * sqrt(min(m,n));
tau = 0.5*sigma * sqrt(log(min(m,n)));
disp('----- run PRPCA -----');
img_est_6 = PRPCA(noisy_outlier_mixed_img, lambda, tau);
RMSE(6) = norm(img_est_6 - img, 'fro') / sqrt(m * n);
PSNR(6) = 10 * log10( 1 / RMSE(6)^2 );

% 5. the proposed smoothed low-rank matrix factorization (sLRMF)
rank = 150;
lambda = sigma * sqrt(min(m,n));
tau = 0.5*sigma * sqrt(log(min(m,n)));
lr = 0.001;
maxIter = 2000;
tol = 1e-5;
disp('----- run proposed sLRMF -----');
img_est_7 = sLRMF_SRPCA(noisy_outlier_mixed_img, rank, lambda, tau, lr, maxIter,tol);
RMSE(7) = norm(img_est_7 - img, 'fro') / sqrt(m * n);
PSNR(7) = 10 * log10( 1 / RMSE(7)^2 );

% compute the RMSE and PSNR of the observation
RMSE(8) = norm(noisy_outlier_mixed_img - img, 'fro') / sqrt(m * n);
PSNR(8) = 10 * log10( 1 / RMSE(8)^2 );

%% results display
fprintf('\n');
fprintf('========================================= Results ============================================\n');
fprintf('  %-10s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  \n', ...
    'Method', 'Obs.', 'NN', 'LRMF', 'TV', 'NN+TV', 'CTV', 'PRPCA', 'sLRMF');
fprintf('  %-8s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  \n', ...
    'RMSE', RMSE(8), RMSE(1), RMSE(2), RMSE(3), RMSE(4), RMSE(5), RMSE(6), RMSE(7));
fprintf('  %-8s  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  \n', ...
    'PSNR', PSNR(8), PSNR(1), PSNR(2), PSNR(3), PSNR(4), PSNR(5), PSNR(6), PSNR(7));
fprintf('========================================= Results ============================================\n');

%% visualization
figure(1)
subplot 331; imshow(img); title('original');
subplot 332; imshow(noisy_outlier_mixed_img); title(['observation ' num2str(RMSE(8))]);
subplot 333; imshow(img_est_1); title(['NN ' num2str(RMSE(1))]);
subplot 334; imshow(img_est_2); title(['LRMF ' num2str(RMSE(2))]);
subplot 335; imshow(img_est_3); title(['TV ' num2str(RMSE(3))]);
subplot 336; imshow(img_est_4); title(['NN+TV ' num2str(RMSE(4))]);
subplot 337; imshow(img_est_5); title(['CTV ' num2str(RMSE(5))]);
subplot 338; imshow(img_est_6); title(['PRPCA ' num2str(RMSE(6))]);
subplot 339; imshow(img_est_7); title(['sLRMF ' num2str(RMSE(7))]);

