%% simulation verification of the linear dependency relationship 
%% between the estimation errors and the smoothness level 'sqrt(beta)'
%% for sLRMF-based stable matrix sensing
% LRMF as a baseline
% Running the following program will take approximately several seconds.

clear;close all;
addpath(genpath(pwd))

%% experiment settings
num_reps = 1; % number of repeated experiments; or 20 for better results

n = 100; 
dim = [n, n];
r = 1; % or r = 5;
sigma = 0.01; 

sample_ratio = 0.2; 
A = PermuteWHT_partitioned(n,n,sample_ratio); 
num_samples = length(A*randn(n,n)); % number of samples
% A = randn(num_samples , n^2)/sqrt(num_samples );

lr = 0.001;
maxIter = 2000; 
tol = 1e-6;

sqrt_beta_range = 0.1: 0.1: 1.9;
num_cases = length(sqrt_beta_range);
beta = zeros(1,num_cases);

Error_Spec_LRMF_SMS = zeros(1,num_cases);
Error_Fro_LRMF_SMS = zeros(1,num_cases);
Error_Inf_LRMF_SMS = zeros(1,num_cases);

Error_Spec_sLRMF_SMS = zeros(1,num_cases);
Error_Fro_sLRMF_SMS = zeros(1,num_cases);
Error_Inf_sLRMF_SMS = zeros(1,num_cases);

for i = 1:num_cases
    for j = 1:num_reps
    disp(['i = ' num2str(i) ', j = ' num2str(j) ]);
    
    input_beta = sqrt_beta_range(i)^2;
    U0 = generate_smooth_subspace(n, r, input_beta);
    V0 = generate_smooth_subspace(n, r, input_beta);
    [~,S0,~] = svds(randn(n,n), r); 
    M0 = U0 * S0 * V0';
    beta(i) = beta(i) + subspace_smoothness_parameter(M0, r);
   
    y = A*M0(:);
    Noisy_y = y + sigma * randn(num_samples,1);
    
    %% LRMF-based stable matrix sensing
    lambda_LRMF = 5 * sigma * sqrt(n);
    M_hat_LRMF = LRMF_SMS(Noisy_y, A, dim, r, lambda_LRMF, lr, maxIter);
    Error_Spec_LRMF_SMS(i) = Error_Spec_LRMF_SMS(i) + norm(M_hat_LRMF - M0) / norm(M0);
    Error_Fro_LRMF_SMS(i) = Error_Fro_LRMF_SMS(i) + norm(M_hat_LRMF - M0, 'fro') / norm(M0, 'fro');
    Error_Inf_LRMF_SMS(i) = Error_Inf_LRMF_SMS(i) + (norm(M_hat_LRMF(:) - M0(:), Inf)) / norm(M0(:), Inf);
    
    %% sLRMF-based stable matrix sensing
    lambda_sLRMF = 5 * sigma * sqrt(n);
    M_hat_sLRMF = sLRMF_SMS(Noisy_y, A, dim, r, lambda_sLRMF, lr, maxIter);
    Error_Spec_sLRMF_SMS(i) = Error_Spec_sLRMF_SMS(i) + norm(M_hat_sLRMF - M0) / norm(M0);
    Error_Fro_sLRMF_SMS(i) = Error_Fro_sLRMF_SMS(i) + norm(M_hat_sLRMF - M0, 'fro') / norm(M0, 'fro');
    Error_Inf_sLRMF_SMS(i) = Error_Inf_sLRMF_SMS(i) + (norm(M_hat_sLRMF(:) - M0(:), Inf))/norm(M0(:), Inf);
    end
end
beta = beta / num_reps;
Error_Spec_LRMF_SMS = Error_Spec_LRMF_SMS / num_reps;
Error_Spec_sLRMF_SMS = Error_Spec_sLRMF_SMS / num_reps;
Error_Fro_LRMF_SMS = Error_Fro_LRMF_SMS / num_reps;
Error_Fro_sLRMF_SMS = Error_Fro_sLRMF_SMS / num_reps;
Error_Inf_LRMF_SMS = Error_Inf_LRMF_SMS / num_reps;
Error_Inf_sLRMF_SMS = Error_Inf_sLRMF_SMS / num_reps;

%% results display
set(0, 'DefaultLineLineWidth', 1); 
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontWeight', 'bold'); 
set(0, 'DefaultAxesTitleFontWeight', 'bold');  

figure(1)
subplot 131
plot(sqrt(beta), Error_Spec_LRMF_SMS, 'bO');hold on;
plot(sqrt(beta), Error_Spec_sLRMF_SMS, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert}{\left\Vert\mathbf{M}_\star\right\Vert}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix sensing');
grid on; grid minor;
subplot 132
plot(sqrt(beta), Error_Fro_LRMF_SMS, 'bO');hold on;
plot(sqrt(beta), Error_Fro_sLRMF_SMS, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}}{\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix sensing');
grid on; grid minor;
subplot 133
plot(sqrt(beta), Error_Inf_LRMF_SMS, 'bO');hold on;
plot(sqrt(beta), Error_Inf_sLRMF_SMS, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\infty}{\left\Vert\mathbf{M}_\star\right\Vert_\infty}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix sensing');
grid on; grid minor;
set(gcf, 'Position', [600, 600, 750, 200]);