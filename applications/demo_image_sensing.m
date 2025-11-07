%%%%% test the low-rank and/or smooth based matrix estimation methods for image compressed sensing task %%%%%
% including: 
% 1) nuclear norm (NN)
% 2) low-rank matrix factorization (LRMF)
% 3) total variation (TV)
% 4) joint nuclear norm and total variation (NN+TV)
% 5) *** the proposed smoothed low-rank matrix factorization (sLRMF) ***

% note that this demo would be time-consuming for NN, TV, NN+TV methods

close all; clear;clc;
addpath(genpath('./lib'));
addpath(genpath('./methods/sensing/'));

% test images from the BSD300 dataset
imgDir = './BSD300-images';
D= dir(fullfile(imgDir,'*.jpg'));

id_of_img = 1; 

img_name = fullfile(imgDir,D(id_of_img).name);
img = double(imread(img_name))/255;
[m, n] = size(img); 
dim = [m, n];

% experiment settings
sigma = 0.1;
sample_ratio = 0.75;
A = PermuteWHT_partitioned(m,n,sample_ratio);
y = A*img(:);
num = length(y);
noisy_y = y + sigma * randn(num, 1);

% evaluate the results of each method by RMSE and PSNR
results_RMSE = zeros(1,5);
results_PSNR = zeros(1,5);

%% run all methods 
% 1. nuclear norm (NN)
gamma = sigma * sqrt(min(m,n));
disp('----- run NN -----');
img_est_1 = NN_SMS(noisy_y, A, dim, gamma);
RMSE(1) = norm(img_est_1 - img, 'fro') / sqrt(m * n);
PSNR(1) = 10 * log10( 1 / RMSE(1)^2 );

% 2. low-rank matrix factorization (LRMF)
rank = 150;
lambda = sigma * sqrt(min(m,n));
lr = 0.001;
maxIter = 1000;
tol = 1e-5;
disp('----- run LRMF -----');
img_est_2 = LRMF_SMS(noisy_y, A, dim, rank, lambda, lr, maxIter, tol);
RMSE(2) = norm(img_est_2 - img, 'fro') / sqrt(m * n);
PSNR(2) = 10 * log10( 1 / RMSE(2)^2 );

% 3. total variation (TV)
beta = sigma / sqrt(min(m,n));
disp('----- run TV -----');
img_est_3 = TV_SMS(noisy_y, A, dim, beta);
RMSE(3) = norm(img_est_3 - img, 'fro') / sqrt(m * n);
PSNR(3) = 10 * log10( 1 / RMSE(3)^2 );

% 4. joint nuclear norm and total variation (NN+TV)
gamma = sigma * sqrt(min(m,n));
beta = sigma / sqrt(min(m,n));
disp('----- run NN+TV -----');
img_est_4 = NN_TV_SMS(noisy_y, A, dim, gamma, beta);
RMSE(4) = norm(img_est_4 - img, 'fro') / sqrt(m * n);
PSNR(4) = 10 * log10( 1 / RMSE(4)^2 );
	
% 5. the proposed smoothed low-rank matrix factorization (sLRMF)
rank = 150;
lambda = sigma * sqrt(min(m,n));
lr = 0.001;
maxIter = 1000;
tol = 1e-5;
disp('----- run proposed sLRMF -----');
img_est_5 = sLRMF_SMS(noisy_y, A, dim, rank, lambda, lr, maxIter, tol);
RMSE(5) = norm(img_est_5 - img, 'fro') / sqrt(m * n);
PSNR(5) = 10 * log10( 1 / RMSE(5)^2 );

%% results display
fprintf('\n');
fprintf('========================================= Results ============================================\n');
fprintf('  %-10s  %-8s  %-8s  %-8s  %-8s  %-8s  \n', ...
    'Method', 'NN', 'LRMF', 'TV', 'NN+TV', 'sLRMF');
fprintf('  %-8s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  \n', ...
    'RMSE', RMSE(1), RMSE(2), RMSE(3), RMSE(4), RMSE(5));
fprintf('  %-8s  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  \n', ...
    'PSNR', PSNR(1), PSNR(2), PSNR(3), PSNR(4), PSNR(5));
fprintf('========================================= Results ============================================\n');

%% visualization
figure(1)
subplot 231; imshow(img); title('original');
subplot 232; imshow(img_est_1); title(['NN ' num2str(RMSE(1))]);
subplot 233; imshow(img_est_2); title(['LRMF ' num2str(RMSE(2))]);
subplot 234; imshow(img_est_3); title(['TV ' num2str(RMSE(3))]);
subplot 235; imshow(img_est_4); title(['NN+TV ' num2str(RMSE(4))]);
subplot 233; imshow(img_est_5); title(['sLRMF ' num2str(RMSE(5))]);
