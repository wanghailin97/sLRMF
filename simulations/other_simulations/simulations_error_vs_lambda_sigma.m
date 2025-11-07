%% simulation experiments
% the estimation errors versus regularization parameter 'lambda' selection 
% among a seris of noise cases
% for sLRMF (LRMF as a beseline)

clear;close all;
addpath(genpath(pwd))

%% synthesis simultaneous low-rank and smooth matrix
n = 100;
r = 5;

input_beta = 0.01;
U0 = generate_smooth_subspace(n, r, input_beta);
V0 = generate_smooth_subspace(n, r, input_beta);
[~,S0,~] = svds(randn(n,n), r);
M0 = U0 * S0 * V0';
beta = subspace_smoothness_parameter(M0, r);

%% experiment settings
lr = 0.001;
maxIter = 5000; 
tol = 1e-8;

sigma_set = [0.01:0.01:0.1 0.2:0.1:1 2:1:10];
num__cases_sigma = length(sigma_set);

coeff = [0 0.05:0.05:1 2:1:20];
num_cases_lambda = length(coeff);

Error_Fro_LRMF_MD = zeros(num__cases_sigma, num_cases_lambda);
Error_Fro_sLRMF_MD = zeros(num__cases_sigma, num_cases_lambda);

for i = 1:num__cases_sigma
    sigma = sigma_set(i);
    Noisy_M = M0 + sigma * randn(n,n);
    
    for j = 1:num_cases_lambda
        
    lambda = coeff(j) * sigma * sqrt(n);
    
    M_hat_LRMF = LRMF_MD(Noisy_M, r, lambda, lr, maxIter, tol);
    Error_Fro_LRMF_MD(i,j) = norm(M_hat_LRMF-M0, 'fro')  / norm(M0,'fro');
    
    M_hat_sLRMF = sLRMF_MD(Noisy_M, r, lambda, lr, maxIter, tol);
    Error_Fro_sLRMF_MD(i,j) = norm(M_hat_sLRMF-M0, 'fro') / norm(M0,'fro');
    end
end

% sets relative errors larger than 1 to 1
Error_Fro_LRMF_MD(isnan(Error_Fro_LRMF_MD)) = max(Error_Fro_LRMF_MD(:), [], 'omitnan'); 
Error_Fro_sLRMF_MD(isnan(Error_Fro_sLRMF_MD)) = max(Error_Fro_sLRMF_MD(:), [], 'omitnan'); 
Error_Fro_LRMF_MD(Error_Fro_LRMF_MD>1) = 1; 
Error_Fro_sLRMF_MD(Error_Fro_sLRMF_MD>1) = 1; 

% obtain the smallest error among all 'lambda' for each noise case
rowMins_Error_Fro_LRMF_MD = min(Error_Fro_LRMF_MD, [], 2);
rowMins_Error_Fro_sLRMF_MD = min(Error_Fro_sLRMF_MD, [], 2);

%% results display
set(0, 'DefaultLineLineWidth', 1);
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultAxesTitleFontWeight', 'bold');

figure(1)
subplot 121
imagesc(flipud(Error_Fro_LRMF_MD));
set(gca,'Xtick',[1 4:4:40]);
X = {'0.05','0.2','0.4','0.6','0.8','1','20','40','60','80','100'};
set(gca,'XTickLabel', X);
set(gca,'Ytick',[1 6 10 15 19 24 28]);
Y = {'10','5','1','0.5','0.1','0.05','0.01'};
set(gca,'YTickLabel', Y);
set(gcf,'color','w')
xlabel('$\lambda/(\sigma\sqrt{n})$','interpreter','latex');
ylabel('$\sigma$','interpreter','Latex');
title('LRMF');
colormap('summer'); colorbar;
subplot 122
imagesc(flipud(Error_Fro_sLRMF_MD));
set(gca,'Xtick',[1 4:4:40]);
X = {'0.05','0.2','0.4','0.6','0.8','1','20','40','60','80','100'};
set(gca,'XTickLabel', X);
set(gca,'Ytick',[1 6 10 15 19 24 28]);
Y = {'10','5','1','0.5','0.1','0.05','0.01'};
set(gca,'YTickLabel', Y);
set(gcf,'color','w')
xlabel('$\lambda/(\sigma\sqrt{n})$','interpreter','latex');
ylabel('$\sigma$','interpreter','Latex');
title('sLRMF');
colormap('summer'); colorbar;
set(gcf, 'Position', [400, 100, 1000, 350]);

figure(2)
subplot 121
plot(rowMins_Error_Fro_LRMF_MD, '-bO');hold on;
plot(rowMins_Error_Fro_sLRMF_MD, '-rp');
set(gca,'Xtick',[1 5 10 14 19 23 28]);
X = {'0.01', '0.05', '0.1', '0.5', '1', '5', '10'};
set(gca,'XTickLabel', X);
xlabel('$\sigma$','interpreter','Latex');
ylabel('$\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}/\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}$','interpreter','Latex');
legend('LRMF', 'sLRMF','location','best');
title('smallest error over all $\lambda$', 'interpret', 'latex');
grid minor; grid on;
subplot 122
plot(rowMins_Error_Fro_sLRMF_MD./rowMins_Error_Fro_LRMF_MD, '-r');
ylim([0 1.1]);
set(gca,'Xtick',[1 5 10 14 19 23 28]);
X = {'0.01', '0.05', '0.1', '0.5', '1', '5', '10'};
set(gca,'XTickLabel', X);
xlabel('$\sigma$','interpreter','Latex');
ylabel('$\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}/\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}$','interpreter','Latex');
legend('error ratio (sLRMF/LRMF)','location','best');
title('ratio of smallest error', 'interpret', 'latex');
grid minor; grid on;

