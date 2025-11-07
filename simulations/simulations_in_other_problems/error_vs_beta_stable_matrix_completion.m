%% simulation verification of the linear dependency relationship 
%% between the estimation errors and the smoothness level 'sqrt(beta)'
%% for sLRMF-based stable matrix completion
% LRMF as a baseline
% Running the following program will take approximately several seconds.

clear;close all;
addpath(genpath(pwd))

%% experiment settings
num_reps = 1; % number of repeated experiments; or 20 for better results

n = 100; 
r = 1; % or r = 5;
sigma = 0.01; 

p = 0.2; % sampling probability

lr = 0.001;
maxIter = 2000; 
tol = 1e-6;

sqrt_beta_range = 0.1: 0.1: 1.9;
num_cases = length(sqrt_beta_range);
beta = zeros(1, num_cases);

Error_Spec_LRMF_SMC = zeros(1, num_cases);
Error_Fro_LRMF_SMC = zeros(1, num_cases);
Error_Inf_LRMF_SMC = zeros(1, num_cases);

Error_Spec_sLRMF_SMC = zeros(1, num_cases);
Error_Fro_sLRMF_SMC = zeros(1, num_cases);
Error_Inf_sLRMF_SMC = zeros(1, num_cases);

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
    Omega = randperm(n^2, round(p*n^2));
    Mask = zeros(n,n); Mask(Omega) = 1;
    
    Noisy_incompleted_M = Noisy_M .* Mask;
    
    %% LRMF-based stable matrix completion
    lambda_LRMF = 5 * sigma * sqrt(n);
    M_hat_LRMF = LRMF_SMC(Noisy_incompleted_M, Mask, r, lambda_LRMF, lr, maxIter);
    Error_Spec_LRMF_SMC(i) = Error_Spec_LRMF_SMC(i) + norm(M_hat_LRMF - M0) / norm(M0);
    Error_Fro_LRMF_SMC(i) = Error_Fro_LRMF_SMC(i) + norm(M_hat_LRMF - M0, 'fro') / norm(M0, 'fro');
    Error_Inf_LRMF_SMC(i) = Error_Inf_LRMF_SMC(i) + (norm(M_hat_LRMF(:) - M0(:), Inf)) / norm(M0(:), Inf);
    
    %% sLRMF-based stable matrix completion
    lambda_sLRMF = 5 * sigma * sqrt(n);
    M_hat_sLRMF = sLRMF_SMC(Noisy_incompleted_M, Mask, r, lambda_sLRMF, lr, maxIter);
    Error_Spec_sLRMF_SMC(i) = Error_Spec_sLRMF_SMC(i) + norm(M_hat_sLRMF - M0) / norm(M0);
    Error_Fro_sLRMF_SMC(i) = Error_Fro_sLRMF_SMC(i) + norm(M_hat_sLRMF - M0, 'fro') / norm(M0, 'fro');
    Error_Inf_sLRMF_SMC(i) = Error_Inf_sLRMF_SMC(i) + (norm(M_hat_sLRMF(:) - M0(:), Inf)) / norm(M0(:), Inf);
    end
end
beta = beta / num_reps;
Error_Spec_LRMF_SMC = Error_Spec_LRMF_SMC / num_reps;
Error_Spec_sLRMF_SMC = Error_Spec_sLRMF_SMC / num_reps;
Error_Fro_LRMF_SMC = Error_Fro_LRMF_SMC / num_reps;
Error_Fro_sLRMF_SMC = Error_Fro_sLRMF_SMC / num_reps;
Error_Inf_LRMF_SMC = Error_Inf_LRMF_SMC / num_reps;
Error_Inf_sLRMF_SMC = Error_Inf_sLRMF_SMC / num_reps;


%% results display
set(0, 'DefaultLineLineWidth', 1); 
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontWeight', 'bold'); 
set(0, 'DefaultAxesTitleFontWeight', 'bold');  

figure(1)
subplot 131
plot(sqrt(beta), Error_Spec_LRMF_SMC, 'bO');hold on;
plot(sqrt(beta), Error_Spec_sLRMF_SMC, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert}{\left\Vert\mathbf{M}_\star\right\Vert}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix completion');
grid on; grid minor;
subplot 132
plot(sqrt(beta), Error_Fro_LRMF_SMC, 'bO');hold on;
plot(sqrt(beta), Error_Fro_sLRMF_SMC, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}}{\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix completion');
grid on; grid minor;
subplot 133
plot(sqrt(beta), Error_Inf_LRMF_SMC, 'bO');hold on;
plot(sqrt(beta), Error_Inf_sLRMF_SMC, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\infty}{\left\Vert\mathbf{M}_\star\right\Vert_\infty}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix completion');
grid on; grid minor;
set(gcf, 'Position', [600, 600, 750, 200]);

