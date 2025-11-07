%% simulation experiments
% the relationship between the estimation errors and the smoothness level 'sqrt(beta)'
% for a seris of low-rank and/or smoothe regularized methods
% including NN, TV, NN+TV. CTV, LRMF and proposed sLRMF


clear;close all;
addpath(genpath(pwd))
addpath(genpath('.\application\methods\matrix denoising'));

%% experiment settings
num_reps = 1; % number of repeated experiments; or 20 for better results

n = 100;
r = 1; % or r = 5;
sigma = 0.1;

lr = 0.001;
maxIter = 2000;
tol = 1e-6;

rho = 1.25;
mu = 0.01;
max_mu = 1e8;

sqrt_beta_range = 0.1: 0.1 :1.9;
num_cases = length(sqrt_beta_range);
beta = zeros(1, num_cases);

Error_Fro_NN = zeros(1,num_cases);
Error_Fro_TV = zeros(1,num_cases);
Error_Fro_NN_TV = zeros(1,num_cases);
Error_Fro_CTV = zeros(1,num_cases);
Error_Fro_LRMF = zeros(1,num_cases);
Error_Fro_sLRMF = zeros(1,num_cases);


for i = 1:num_cases
    for j = 1:num_reps
        disp(['i = ' num2str(i) ', j = ' num2str(j) ]);
        
        input_beta = sqrt_beta_range(i)^2;
        U0 = generate_smooth_subspace(n, r, input_beta);
        V0 = generate_smooth_subspace(n, r, input_beta);
        [~,S0,~] = svds(randn(n,n), r);
        M0 = U0 * S0 * V0';
        beta(i) = beta(i) + subspace_smoothness_parameter(M0, r);
        
        Noisy_M = M0 + sigma * randn(n,n);
        
        M_hat_NN  = NN_MD(Noisy_M, 5 * sigma * sqrt(n));
        Error_Fro_NN(i) = Error_Fro_NN(i) + norm(M_hat_NN-M0, 'fro') / norm(M0, 'fro');
        
        M_hat_LRMF = LRMF_MD(Noisy_M, r, 5 * sigma * sqrt(n), lr, maxIter, tol);
        Error_Fro_LRMF(i) = Error_Fro_LRMF(i) + norm(M_hat_LRMF - M0, 'fro') / norm(M0, 'fro');
        
        M_hat_TV = TV_MD(Noisy_M, 5 * sigma / sqrt(n), rho);
        Error_Fro_TV(i) = Error_Fro_TV(i) + norm(M_hat_TV-M0, 'fro') / norm(M0, 'fro');
        
        M_hat_NN_TV = NN_TV_MD(Noisy_M, 2 * sigma * sqrt(n), 10 * sigma / sqrt(n), rho);
        Error_Fro_NN_TV(i) = Error_Fro_NN_TV(i) + norm(M_hat_NN_TV-M0, 'fro') / norm(M0, 'fro');
        
        M_hat_CTV = CTV_MD(Noisy_M, 3 * sigma * sqrt(n), rho);
        Error_Fro_CTV(i) = Error_Fro_CTV(i) + norm(M_hat_CTV-M0, 'fro') / norm(M0, 'fro');
        
        M_hat_sLRMF = sLRMF_MD(Noisy_M, r, 5 * sigma * sqrt(n), lr, maxIter, tol);
        Error_Fro_sLRMF(i) = Error_Fro_sLRMF(i) + norm(M_hat_sLRMF-M0, 'fro') / norm(M0, 'fro');
    end
end
beta = beta / num_reps;
Error_Fro_NN = Error_Fro_NN/num_reps;
Error_Fro_TV = Error_Fro_TV/num_reps;
Error_Fro_NN_TV = Error_Fro_NN_TV/num_reps;
Error_Fro_CTV = Error_Fro_CTV/num_reps;
Error_Fro_LRMF = Error_Fro_LRMF/num_reps;
Error_Fro_sLRMF = Error_Fro_sLRMF/num_reps;

%% results display
set(0, 'DefaultLineLineWidth', 1);
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultAxesTitleFontWeight', 'bold');

figure(1)
plot(sqrt(beta), Error_Fro_NN, 'k+',  'MarkerSize', 8);hold on;
plot(sqrt(beta), Error_Fro_TV, 'gx',  'MarkerSize', 8);hold on;
plot(sqrt(beta), Error_Fro_NN_TV, 'cs', 'MarkerSize', 8);hold on;
plot(sqrt(beta), Error_Fro_CTV, 'mh', 'MarkerSize', 8);hold on;
plot(sqrt(beta), Error_Fro_LRMF, 'bo', 'MarkerSize', 8);hold on;
plot(sqrt(beta), Error_Fro_sLRMF, 'rp', 'MarkerSize', 10);hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\left\Vert\hat{\mathbf{M}}-\mathbf{M}_\star\right\Vert_\mathrm{F}/\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}$','interpreter','Latex');
legend('NN', 'TV', 'NN+TV', 'CTV', 'LRMF', 'sLRMF', 'location', 'northwest');
grid on; grid minor;
%set(gcf, 'Position', [200, 500, 250, 400]);
