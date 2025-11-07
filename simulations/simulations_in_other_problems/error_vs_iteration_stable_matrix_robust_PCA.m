%% simulation verification the linear convergence behaviour
%% of the gradient descent algroithm with spectral initialization 
%% for sLRMF-based stable matrix robust PCA
% LRMF as a beseline
% with spectral intinalization (SI) and smooth spectral intinalization (SSI)

clear; close all;
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
sigma = 0.1;
lr = 0.005;
outlier_ratio = 0.2; % rato of sparse outliers
Omega_s = randperm(n^2, round(outlier_ratio*n^2));
Outlier_all = 5*randn(n,n);
Outlier = zeros(n, n);
Outlier(Omega_s) = Outlier_all(Omega_s);

Obs = M0 + Outlier + sigma * randn(n,n);

num_iter = 500;

Error_Spec_LRMF_SRPCA_SI = zeros(1,num_iter);
Error_Fro_LRMF_SRPCA_SI = zeros(1,num_iter);
Error_Inf_LRMF_SRPCA_SI = zeros(1,num_iter);

Error_Spec_sLRMF_SRPCA_SI = zeros(1,num_iter);
Error_Fro_sLRMF_SRPCA_SI = zeros(1,num_iter);
Error_Inf_sLRMF_SRPCA_SI = zeros(1,num_iter);

Error_Spec_sLRMF_SRPCA_SSI = zeros(1,num_iter);
Error_Fro_sLRMF_SRPCA_SSI = zeros(1,num_iter);
Error_Inf_sLRMF_SRPCA_SSI = zeros(1,num_iter);

Error_Spec_Outlier_LRMF_SRPCA_SI = zeros(1,num_iter);
Error_Spec_Outlier_sLRMF_SRPCA_SI = zeros(1,num_iter);
Error_Spec_Outlier_sLRMF_SRPCA_SSI = zeros(1,num_iter);

Error_Inf_Outlier_LRMF_SRPCA_SI = zeros(1,num_iter);
Error_Inf_Outlier_sLRMF_SRPCA_SI = zeros(1,num_iter);
Error_Inf_Outlier_sLRMF_SRPCA_SSI = zeros(1,num_iter);


%% run LRMF-based matrix robust PCA with spectral intinalization
lambda = 5 * sigma * sqrt(n); 
tau = 2 * sigma * sqrt(log(n));
% spectral initialization
Tim_Obs = trimming(Obs, max(M0(:))+sigma*log(n));
[U0,S0,V0] = svds(Tim_Obs, r);
X = U0*sqrt(S0);
Y = V0*sqrt(S0);
O = soft_thre(Obs - X*Y', tau);
% main
for iter = 1:num_iter
    Xk = X; Yk = Y; XkYkt = Xk*Yk';
    iterate = Xk*Yk';
    
    Error_Spec_LRMF_SRPCA_SI(iter) = norm(iterate-M0)/norm(M0(:));
    Error_Fro_LRMF_SRPCA_SI(iter) = norm(iterate-M0, 'fro')/norm(M0,'fro');
    Error_Inf_LRMF_SRPCA_SI(iter) = norm(iterate(:)-M0(:), Inf)/norm(M0(:),Inf);
    Error_Spec_Outlier_LRMF_SRPCA_SI(iter) = norm(O-Outlier)/norm(Outlier);
    Error_Inf_Outlier_LRMF_SRPCA_SI(iter) = norm(O(:)-Outlier(:), Inf)/norm(Outlier(:), Inf);
    
    Grad_Xk = (Xk*Yk' + O - Obs) * Yk + lambda * Xk + 0.5 * Xk * (Xk'*Xk-Yk'*Yk);
    X = Xk - lr*Grad_Xk;
    Grad_Yk =  (Xk*Yk' + O - Obs)' * Xk + lambda * Yk + 0.5 * Yk * (Yk'*Yk-Xk'*Xk);
    Y = Yk - lr*Grad_Yk;
    O = soft_thre(Obs - X*Y', tau);
end

%% run sLRMF-based matrix robust PCA with spectral intinalization
lambda = 5 * sigma * sqrt(n); 
tau = 2 * sigma * sqrt(log(n));
% spectral initialization
Tim_Obs = trimming(Obs, max(M0(:))+sigma*log(n));
[U0,S0,V0] = svds(Tim_Obs, r);
X = U0*sqrt(S0);
Y = V0*sqrt(S0);
O = soft_thre(Obs - X*Y', tau);
% main
for iter = 1:num_iter
    Xk = X; Yk = Y; XkYkt = Xk*Yk';
    iterate = Xk*Yk';
    
    Error_Spec_sLRMF_SRPCA_SI(iter) = norm(iterate-M0)/norm(M0(:));
    Error_Fro_sLRMF_SRPCA_SI(iter) = norm(iterate-M0, 'fro')/norm(M0,'fro');
    Error_Inf_sLRMF_SRPCA_SI(iter) = norm(iterate(:)-M0(:), Inf)/norm(M0(:),Inf);
    Error_Spec_Outlier_sLRMF_SRPCA_SI(iter) = norm(O-Outlier)/norm(Outlier);
    Error_Inf_Outlier_sLRMF_SRPCA_SI(iter) = norm(O(:)-Outlier(:), Inf)/norm(Outlier(:), Inf);
    
    Grad_Xk = (Xk*Yk' + O - Obs) * Yk + lambda * DiffT(Diff(Xk)) + 0.5 * Xk * (Xk'*Xk-Yk'*Yk);
    X = Xk - lr*Grad_Xk;
    Grad_Yk =  (Xk*Yk' + O - Obs)' * Xk + lambda * DiffT(Diff(Yk)) + 0.5 * Yk * (Yk'*Yk-Xk'*Xk);
    Y = Yk - lr*Grad_Yk;
    O = soft_thre(Obs - X*Y', tau);
end

%% run sLRMF-based matrix robust PCA with smoothed spectral intinalization
lambda = 5 * sigma * sqrt(n); 
tau = 2 * sigma * sqrt(log(n));
gamma = 0.5*lambda;
% spectral initialization
Tim_Obs = trimming(Obs, max(M0(:))+sigma*log(n));
[X, Y] = smoothed_spectral_initialization(Tim_Obs, r, gamma);
O = soft_thre(Obs - X*Y', tau);

% main
for iter = 1:num_iter
    Xk = X; Yk = Y; XkYkt = Xk*Yk';
    iterate = Xk*Yk';
    
    Error_Spec_sLRMF_SRPCA_SSI(iter) = norm(iterate-M0)/norm(M0(:));
    Error_Fro_sLRMF_SRPCA_SSI(iter) = norm(iterate-M0, 'fro')/norm(M0,'fro');
    Error_Inf_sLRMF_SRPCA_SSI(iter) = norm(iterate(:)-M0(:), Inf)/norm(M0(:),Inf);
    Error_Spec_Outlier_sLRMF_SRPCA_SSI(iter) = norm(O-Outlier)/norm(Outlier);
    Error_Inf_Outlier_sLRMF_SRPCA_SSI(iter) = norm(O(:)-Outlier(:), Inf)/norm(Outlier(:), Inf);
    
    Grad_Xk = (Xk*Yk' + O - Obs) * Yk + lambda * DiffT(Diff(Xk)) + 0.5 * Xk * (Xk'*Xk-Yk'*Yk);
    X = Xk - lr*Grad_Xk;
    Grad_Yk =  (Xk*Yk' + O - Obs)' * Xk + lambda * DiffT(Diff(Yk)) + 0.5 * Yk * (Yk'*Yk-Xk'*Xk);
    Y = Yk - lr*Grad_Yk;
    O = soft_thre(Obs - X*Y', tau);
end

%% results display
close all;
set(0, 'DefaultLineLineWidth', 1); % 设置所有文字（标题、标签、图例等）为加粗
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontWeight', 'bold');  % 坐标轴文字
set(0, 'DefaultAxesTitleFontWeight', 'bold');  % 坐标轴文字

iter_show = 500;

figure(1)
subplot 151
plot(1:iter_show, Error_Spec_LRMF_SRPCA_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Spec_sLRMF_SRPCA_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Spec_sLRMF_SRPCA_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert/\left\Vert\mathbf{M}_\star\right\Vert$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 152
plot(1:iter_show, Error_Fro_LRMF_SRPCA_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Fro_sLRMF_SRPCA_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Fro_sLRMF_SRPCA_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}/\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 153
plot(1:iter_show, Error_Inf_LRMF_SRPCA_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Inf_sLRMF_SRPCA_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Inf_sLRMF_SRPCA_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_{\infty}/\left\Vert\mathbf{M}_\star\right\Vert_\infty$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 154
plot(1:iter_show, Error_Spec_Outlier_LRMF_SRPCA_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Spec_Outlier_sLRMF_SRPCA_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Spec_Outlier_sLRMF_SRPCA_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{S}_t-\mathbf{S}_\star\right\Vert/\left\Vert\mathbf{S}_\star\right\Vert$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 155
plot(1:iter_show, Error_Inf_Outlier_LRMF_SRPCA_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Inf_Outlier_sLRMF_SRPCA_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Inf_Outlier_sLRMF_SRPCA_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{S}_t-\mathbf{S}_\star\right\Vert_\infty/\left\Vert\mathbf{S}_\star\right\Vert_\infty$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
set(gcf, 'Position', [600, 600, 1250, 200]);

figure(2)
subplot 151
plot(1:iter_show, log(Error_Spec_LRMF_SRPCA_SI(1:iter_show)-Error_Spec_LRMF_SRPCA_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Spec_sLRMF_SRPCA_SI(1:iter_show)-Error_Spec_sLRMF_SRPCA_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Spec_sLRMF_SRPCA_SSI(1:iter_show)-Error_Spec_sLRMF_SRPCA_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\log\left(\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert-\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert\right)$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 152
plot(1:iter_show, log(Error_Fro_LRMF_SRPCA_SI(1:iter_show)-Error_Fro_LRMF_SRPCA_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Fro_sLRMF_SRPCA_SI(1:iter_show)-Error_Fro_sLRMF_SRPCA_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Fro_sLRMF_SRPCA_SSI(1:iter_show)-Error_Fro_sLRMF_SRPCA_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\log\left(\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}-\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}\right)$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 153
plot(1:iter_show, log(Error_Inf_LRMF_SRPCA_SI(1:iter_show)-Error_Inf_LRMF_SRPCA_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Inf_sLRMF_SRPCA_SI(1:iter_show)-Error_Inf_sLRMF_SRPCA_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Inf_sLRMF_SRPCA_SSI(1:iter_show)-Error_Inf_sLRMF_SRPCA_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\log\left(\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_{\infty}-\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\infty\right)$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 154
plot(1:iter_show, log(Error_Spec_Outlier_LRMF_SRPCA_SI(1:iter_show)-Error_Spec_Outlier_LRMF_SRPCA_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Spec_Outlier_sLRMF_SRPCA_SI(1:iter_show)-Error_Spec_Outlier_sLRMF_SRPCA_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Spec_Outlier_sLRMF_SRPCA_SSI(1:iter_show)-Error_Spec_Outlier_sLRMF_SRPCA_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\log\left(\left\Vert\mathbf{S}_t-\mathbf{S}_\star\right\Vert-\left\Vert\hat{\mathbf{S}}-\mathbf{S}_\star\right\Vert\right)$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
subplot 155
plot(1:iter_show, log(Error_Inf_Outlier_LRMF_SRPCA_SI(1:iter_show)-Error_Inf_Outlier_LRMF_SRPCA_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Inf_Outlier_sLRMF_SRPCA_SI(1:iter_show)-Error_Inf_Outlier_sLRMF_SRPCA_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Inf_Outlier_sLRMF_SRPCA_SSI(1:iter_show)-Error_Inf_Outlier_sLRMF_SRPCA_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\log\left(\left\Vert\mathbf{S}_t-\mathbf{S}_\star\right\Vert_\infty-\left\Vert\hat{\mathbf{S}}-\mathbf{S}_\star\right\Vert_\infty\right)$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix robust PCA');
grid on; grid minor;
set(gcf, 'Position', [600, 300, 1250, 200]);
