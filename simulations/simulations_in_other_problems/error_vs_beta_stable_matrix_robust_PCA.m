%% simulation verification of the linear dependency relationship 
%% between the estimation errors and the smoothness level 'sqrt(beta)'
%% for sLRMF-based stable matrix robust PCA
% LRMF as a baseline
% Running the following program will take approximately several seconds.

clear;close all;
addpath(genpath(pwd))

%% experiment settings
num_reps = 1; % number of repeated experiments; or 20 for better results

n = 100; 
dim = [n, n];
r = 1; % or r = 5;
sigma = 0.05; 

outlier_ratio = 0.2; % ration of sparse outliers
Omega_s = randperm(n^2, round(outlier_ratio * n^2));
Outlier_all = 10 * randn(n, n);
Outlier = zeros(n, n);
Outlier(Omega_s) = Outlier_all(Omega_s);

lr = 0.005;
maxIter = 5000; 
tol = 1e-8;

sqrt_beta_range = 0.1: 0.1: 1.9;
num_cases = length(sqrt_beta_range);
beta = zeros(1, num_cases);

Error_Spec_LRMF_SRPCA = zeros(1, num_cases);
Error_Fro_LRMF_SRPCA = zeros(1, num_cases);
Error_Inf_LRMF_SRPCA = zeros(1, num_cases);

Error_Spec_sLRMF_SRPCA = zeros(1, num_cases);
Error_Fro_sLRMF_SRPCA = zeros(1, num_cases);
Error_Inf_sLRMF_SRPCA = zeros(1, num_cases);

Error_Outlier_Spec_LRMF_SRPCA = zeros(1, num_cases);
Error_Outlier_Fro_LRMF_SRPCA = zeros(1, num_cases);
Error_Outlier_Inf_LRMF_SRPCA = zeros(1, num_cases);

Error_Outlier_Spec_sLRMF_SRPCA = zeros(1, num_cases);
Error_Outlier_Fro_sLRMF_SRPCA = zeros(1, num_cases);
Error_Outlier_Inf_sLRMF_SRPCA = zeros(1, num_cases);

for i = 1:num_cases
    for j = 1:num_reps
    disp(['i = ' num2str(i) ', j = ' num2str(j) ]);
    
    input_beta = sqrt_beta_range(i)^2;
    U0 = generate_smooth_subspace(n, r, input_beta);
    V0 = generate_smooth_subspace(n, r, input_beta);
    [~,S0,~] = svds(randn(n,n), r); 
    M0 = U0 * S0 * V0';
    beta(i) = beta(i) + subspace_smoothness_parameter(M0, r);
    
    Noisy_Outlier_Mixed_M = M0 + Outlier + sigma * randn(n,n);
    
    %% LRMF-based stable matrix robust PCA
    lambda = 10 * sigma * sqrt(n);
    tau = 2 * sigma * sqrt(log(n)); 
    
    [M_hat_LRMF, S_hat_LRMF,~,~] = LRMF_SRPCA(Noisy_Outlier_Mixed_M, r, lambda, tau, lr, maxIter, tol);
    Error_Spec_LRMF_SRPCA(i) = Error_Spec_LRMF_SRPCA(i) + norm(M_hat_LRMF-M0)/norm(M0);
    Error_Fro_LRMF_SRPCA(i) = Error_Fro_LRMF_SRPCA(i) + norm(M_hat_LRMF-M0, 'fro')/norm(M0,'fro');
    Error_Inf_LRMF_SRPCA(i) = Error_Inf_LRMF_SRPCA(i) + norm(M_hat_LRMF(:)-M0(:), Inf)/norm(M0(:),Inf);
    Error_Outlier_Spec_LRMF_SRPCA(i) = Error_Outlier_Spec_LRMF_SRPCA(i) + norm(S_hat_LRMF-Outlier)/norm(Outlier);
    Error_Outlier_Fro_LRMF_SRPCA(i) = Error_Outlier_Fro_LRMF_SRPCA(i) + norm(S_hat_LRMF-Outlier, 'fro')/norm(Outlier,'fro');
    Error_Outlier_Inf_LRMF_SRPCA(i) = Error_Outlier_Inf_LRMF_SRPCA(i) + norm(S_hat_LRMF(:)-Outlier(:), Inf)/norm(Outlier(:),Inf);
    
    %% sLRMF-based stable matrix robust PCA
    lambda = 10 * sigma * sqrt(n);
    tau = 2 * sigma * sqrt(log(n)); 
    
    [M_hat_sLRMF, S_hat_sLRMF,~,~] = sLRMF_SRPCA(Noisy_Outlier_Mixed_M, r, lambda, tau, lr, maxIter, tol);
    Error_Spec_sLRMF_SRPCA(i) = Error_Spec_sLRMF_SRPCA(i) + norm(M_hat_sLRMF-M0)/norm(M0);
    Error_Fro_sLRMF_SRPCA(i) = Error_Fro_sLRMF_SRPCA(i) + norm(M_hat_sLRMF-M0, 'fro')/norm(M0,'fro');
    Error_Inf_sLRMF_SRPCA(i) = Error_Inf_sLRMF_SRPCA(i) + norm(M_hat_sLRMF(:)-M0(:), Inf)/norm(M0(:),Inf);    
    Error_Outlier_Spec_sLRMF_SRPCA(i) = Error_Outlier_Spec_sLRMF_SRPCA(i) + norm(S_hat_sLRMF-Outlier)/norm(Outlier);
    Error_Outlier_Fro_sLRMF_SRPCA(i) = Error_Outlier_Fro_sLRMF_SRPCA(i) + norm(S_hat_sLRMF-Outlier, 'fro')/norm(Outlier,'fro');
    Error_Outlier_Inf_sLRMF_SRPCA(i) = Error_Outlier_Inf_sLRMF_SRPCA(i) + norm(S_hat_sLRMF(:)-Outlier(:), Inf)/norm(Outlier(:),Inf);
    end
end
beta = beta / num_reps;
Error_Spec_LRMF_SRPCA = Error_Spec_LRMF_SRPCA / num_reps;
Error_Spec_sLRMF_SRPCA = Error_Spec_sLRMF_SRPCA / num_reps;
Error_Fro_LRMF_SRPCA = Error_Fro_LRMF_SRPCA / num_reps;
Error_Fro_sLRMF_SRPCA = Error_Fro_sLRMF_SRPCA / num_reps;
Error_Inf_LRMF_SRPCA = Error_Inf_LRMF_SRPCA / num_reps;
Error_Inf_sLRMF_SRPCA = Error_Inf_sLRMF_SRPCA / num_reps;
Error_Outlier_Spec_LRMF_SRPCA = Error_Outlier_Spec_LRMF_SRPCA / num_reps;
Error_Outlier_Spec_sLRMF_SRPCA = Error_Outlier_Spec_sLRMF_SRPCA / num_reps;
Error_Outlier_Fro_LRMF_SRPCA = Error_Outlier_Fro_LRMF_SRPCA / num_reps;
Error_Outlier_Fro_sLRMF_SRPCA = Error_Outlier_Fro_sLRMF_SRPCA / num_reps;
Error_Outlier_Inf_LRMF_SRPCA = Error_Outlier_Inf_LRMF_SRPCA / num_reps;
Error_Outlier_Inf_sLRMF_SRPCA = Error_Outlier_Inf_sLRMF_SRPCA / num_reps;

%% results display
set(0, 'DefaultLineLineWidth', 1); 
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontWeight', 'bold'); 
set(0, 'DefaultAxesTitleFontWeight', 'bold');  

figure(1)
subplot 151
plot(sqrt(beta), Error_Spec_LRMF_SRPCA, 'bO');hold on;
plot(sqrt(beta), Error_Spec_sLRMF_SRPCA, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert}{\left\Vert\mathbf{M}_\star\right\Vert}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 152
plot(sqrt(beta), Error_Fro_LRMF_SRPCA, 'bO');hold on;
plot(sqrt(beta), Error_Fro_sLRMF_SRPCA, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}}{\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 153
plot(sqrt(beta), Error_Inf_LRMF_SRPCA, 'bO');hold on;
plot(sqrt(beta), Error_Inf_sLRMF_SRPCA, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\frac{\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\infty}{\left\Vert\mathbf{M}_\star\right\Vert_\infty}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 154
plot(sqrt(beta), Error_Outlier_Spec_LRMF_SRPCA, 'bO');hold on;
plot(sqrt(beta), Error_Outlier_Spec_sLRMF_SRPCA, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\left\Vert\hat{\mathbf{S}}-\mathbf{S}_\star\right\Vert/\left\Vert\mathbf{S}_\star\right\Vert$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 155
plot(sqrt(beta), Error_Outlier_Inf_LRMF_SRPCA, 'bO');hold on;
plot(sqrt(beta), Error_Outlier_Inf_sLRMF_SRPCA, 'rp');hold on;
xlabel('$\sqrt{\beta}$','interpreter','Latex');
ylabel('$\left\Vert\hat{\mathbf{S}}-\mathbf{S}_\star\right\Vert_\infty/\left\Vert\mathbf{S}_\star\right\Vert_\infty$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('matrix robust PCA');
grid on; grid minor;
set(gcf, 'Position', [600, 600, 1250, 200]);
