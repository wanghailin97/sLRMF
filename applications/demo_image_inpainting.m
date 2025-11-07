%%%%% test the low-rank and/or smooth based matrix estimation methods for image inpainting task %%%%%
% including: 
% 1) nuclear norm (NN)
% 2) low-rank matrix factorization (LRMF)
% 3) total variation (TV)
% 4) joint nuclear norm and total variation (NN+TV)
% 5) correlated total variation (CTV)
% 6) *** the proposed smoothed low-rank matrix factorization (sLRMF) ***

close all; clear; clc;
addpath(genpath('./lib'));
addpath(genpath('./methods/completion'));

% test images from the BSD300 dataset
imgDir = './BSD300-images';
D= dir(fullfile(imgDir,'*.jpg'));

id_of_img = 2;
img_name = fullfile(imgDir,D(id_of_img).name);
img = double(imread(img_name))/255;
[m, n] = size(img); 

% experiment settings
sigma = 0.05;
line_text = 'line_missing';
noisy_img = img + sigma * randn(m,n);

% evaluate the results of each method by RMSE and PSNR
RMSE = zeros(1,7);
PSNR = zeros(1,7);

%% missing cases
%% 1. random missing 
% clear the following annotation symbols to choose this missing case
% p = 0.5;
% Omega = randperm(m*n, round(p*m*n));
% noisy_incompleted_img = zeros(m,n);
% noisy_incompleted_img(Omega) = noisy_img(Omega);
% Mask = zeros(m,n);
% Mask(Omega) = 1;

%% 2.non-random missing 
% clear the following annotation symbols to choose this missing case
Mask = ones(m,n);
Mask(5:5:310,:) = 0;
Mask(:,5:5:470) = 0;
p = 0.8;
noisy_incompleted_img = noisy_img .* Mask;

%% run all methods 
% 1. nuclear norm (NN)
gamma = sigma * sqrt(min(m,n)/p);
disp('----- run NN -----');
img_est_1 = NN_SMC(noisy_incompleted_img, Mask, gamma);
RMSE(1) = norm(img_est_1 - img, 'fro') / sqrt(m * n);
PSNR(1) = 10 * log10( 1 / RMSE(1)^2 );

% 2. low-rank matrix factorization (LRMF)
rank = 150;
lambda = sigma * sqrt(min(m,n)*p);
lr = 0.001;
maxIter = 2000;
tol = 1e-5;
disp('----- run LRMF -----');
img_est_2 = LRMF_SMC(noisy_incompleted_img, Mask, rank, lambda, lr, maxIter, tol);
RMSE(2) = norm(img_est_2 - img, 'fro') / sqrt(m * n);
PSNR(2) = 10 * log10( 1 / RMSE(2)^2 );

% 3. total variation (TV)
beta = sigma / sqrt(min(m,n)*p);
disp('----- run TV -----');
img_est_3 = TV_SMC(noisy_incompleted_img, Mask, beta);
RMSE(3) = norm(img_est_3 - img, 'fro') / sqrt(m * n);
PSNR(3) = 10 * log10( 1 / RMSE(3)^2 );

% 4. joint nuclear norm and total variation (NN+TV)
gamma = sigma * sqrt(min(m,n)/p);
beta = sigma / sqrt(min(m,n)*p);
disp('----- run NN+TV -----');
img_est_4 = NN_TV_SMC(noisy_incompleted_img, Mask, gamma, beta);
RMSE(4) = norm(img_est_4 - img, 'fro') / sqrt(m * n);
PSNR(4) = 10 * log10( 1 / RMSE(4)^2 );
	
% 5. correlated total variation (CTV)
lambda = sigma * sqrt(min(m,n)/p);
disp('----- run CTV -----');
img_est_5 = CTV_SMC(noisy_incompleted_img, Mask, lambda);
RMSE(5) = norm(img_est_5 - img, 'fro') / sqrt(m * n);
PSNR(5) = 10 * log10( 1 / RMSE(5)^2 );

% 6. the proposed smoothed low-rank matrix factorization (sLRMF)
rank = 150;
lambda = sigma * sqrt(min(m,n)*p);
lr = 0.001;
maxIter = 2000;
tol = 1e-5;
disp('----- run proposed sLRMF -----');
img_est_6 = sLRMF_SMC(noisy_incompleted_img, Mask, rank, lambda, lr, maxIter, tol);
RMSE(6) = norm(img_est_6 - img, 'fro') / sqrt(m * n);
PSNR(6) = 10 * log10( 1 / RMSE(6)^2 );

% compute the RMSE and PSNR of the observation
RMSE(7) = norm(noisy_incompleted_img - img, 'fro') / sqrt(m * n);
PSNR(7) = 10 * log10( 1 / RMSE(7)^2 );

%% results display
fprintf('\n');
fprintf('========================================= Results ============================================\n');
fprintf('  %-10s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  \n', ...
    'Method', 'Obs.', 'NN', 'LRMF', 'TV', 'NN+TV', 'CTV', 'sLRMF');
fprintf('  %-8s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  \n', ...
    'RMSE', RMSE(7), RMSE(1), RMSE(2), RMSE(3), RMSE(4), RMSE(5), RMSE(6));
fprintf('  %-8s  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  \n', ...
    'PSNR', PSNR(7), PSNR(1), PSNR(2), PSNR(3), PSNR(4), PSNR(5), PSNR(6));
fprintf('========================================= Results ============================================\n');

%% visualization
figure(1)
subplot 241; imshow(img); title('original');
subplot 242; imshow(noisy_incompleted_img); title(['observation ' num2str(RMSE(7))]);
subplot 243; imshow(img_est_1); title(['NN ' num2str(RMSE(1))]);
subplot 244; imshow(img_est_2); title(['LRMF ' num2str(RMSE(2))]);
subplot 245; imshow(img_est_3); title(['TV ' num2str(RMSE(3))]);
subplot 246; imshow(img_est_4); title(['NN+TV ' num2str(RMSE(4))]);
subplot 247; imshow(img_est_5); title(['CTV ' num2str(RMSE(5))]);
subplot 248; imshow(img_est_6); title(['sLRMF ' num2str(RMSE(6))]);
