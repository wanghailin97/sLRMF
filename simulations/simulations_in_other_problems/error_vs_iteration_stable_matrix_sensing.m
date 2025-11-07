%% simulation verification the linear convergence behaviour
%% of the gradient descent algroithm with spectral initialization 
%% for sLRMF-based stable matrix sensing
% LRMF as a beseline
% with spectral intinalization (SI) and smooth spectral intinalization (SSI)
% note that the matrix sensing problem could be quite time-consuming

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
lr = 0.001;
sample_ratio = 0.2; 
A = PermuteWHT_partitioned(n,n,sample_ratio); % Walsh-Hadamard sampling operation to prevent memory overflow 
num_samples = length(A*randn(n,n)); % number of samples
A = randn(num_samples , n^2)/sqrt(num_samples);
y = A*M0(:) + sigma * randn(num_samples, 1);

num_iter = 500;
Error_Spec_LRMF_SMS_SI = zeros(1, num_iter);
Error_Fro_LRMF_SMS_SI = zeros(1, num_iter);
Error_Inf_LRMF_SMS_SI = zeros(1, num_iter);

Error_Spec_sLRMF_SMS_SI = zeros(1, num_iter);
Error_Fro_sLRMF_SMS_SI = zeros(1, num_iter);
Error_Inf_sLRMF_SMS_SI = zeros(1, num_iter);

Error_Spec_sLRMF_SMS_SSI = zeros(1, num_iter);
Error_Fro_sLRMF_SMS_SSI = zeros(1, num_iter);
Error_Inf_sLRMF_SMS_SSI = zeros(1, num_iter);

%% run LRMF-based stable matrix sensing with spectral intinalization
lambda = sigma * sqrt(n); 
% spectral initialization
Aty = reshape(A'*y, [n, n]);
[U0,S0,V0] = svds(Aty, r);
X = U0*sqrt(S0);
Y = V0*sqrt(S0);
% main
for iter = 1:num_iter
    Xk = X; Yk = Y; XkYkt = Xk*Yk';
    iterate = Xk*Yk';
    Error_Spec_LRMF_SMS_SI(iter) = norm(iterate-M0) / norm(M0(:));
    Error_Fro_LRMF_SMS_SI(iter) = norm(iterate-M0, 'fro') / norm(M0,'fro');
    Error_Inf_LRMF_SMS_SI(iter) = norm(iterate(:)-M0(:), Inf) / norm(M0(:),Inf);
    Grad_Xk = (reshape(A'*(A*XkYkt(:)), [n,n]) - Aty) * Yk + lambda * Xk + 0.5 * Xk * (Xk'*Xk-Yk'*Yk);
    X = Xk - lr*Grad_Xk;
    Grad_Yk = (reshape(A'*(A*XkYkt(:)), [n,n]) - Aty)' * Xk + lambda * Yk + 0.5 * Yk * (Yk'*Yk-Xk'*Xk);
    Y = Yk - lr*Grad_Yk;
end

%% run sLRMF-based stable matrix sensing with spectral intinalization
lambda = 5*sigma * sqrt(n); 
% spectral initialization
Aty = reshape(A'*y, [n, n]);
[U0,S0,V0] = svds(Aty, r);
X = U0*sqrt(S0);
Y = V0*sqrt(S0);
% main
for iter = 1:num_iter
    Xk = X; Yk = Y; XkYkt = Xk*Yk';
    iterate = Xk*Yk';
    Error_Spec_sLRMF_SMS_SI(iter) = norm(iterate-M0) / norm(M0(:));
    Error_Fro_sLRMF_SMS_SI(iter) = norm(iterate-M0, 'fro') / norm(M0,'fro');
    Error_Inf_sLRMF_SMS_SI(iter) = norm(iterate(:)-M0(:), Inf) / norm(M0(:),Inf);
    Grad_Xk = (reshape(A'*(A*XkYkt(:)), [n,n]) - Aty) * Yk + lambda * DiffT(Diff(Xk)) + 0.5 * Xk * (Xk'*Xk-Yk'*Yk);
    X = Xk - lr*Grad_Xk;
    Grad_Yk = (reshape(A'*(A*XkYkt(:)), [n,n]) - Aty)' * Xk + lambda * DiffT(Diff(Yk)) + 0.5 * Yk * (Yk'*Yk-Xk'*Xk);
    Y = Yk - lr*Grad_Yk;
end

%% run sLRMF-based stable matrix sensing with smoothed spectral intinalization
lambda = 5*sigma * sqrt(n); 
gamma = lambda;
% smooth spectral initialization
Aty = reshape(A'*y, [n, n]);
[X, Y] = smoothed_spectral_initialization(Aty, r, gamma);

% main
for iter = 1:num_iter
    Xk = X; Yk = Y; XkYkt = Xk*Yk';
    iterate = Xk*Yk';
    Error_Spec_sLRMF_SMS_SSI(iter) = norm(iterate-M0) / norm(M0(:));
    Error_Fro_sLRMF_SMS_SSI(iter) = norm(iterate-M0, 'fro') / norm(M0,'fro');
    Error_Inf_sLRMF_SMS_SSI(iter) = norm(iterate(:)-M0(:), Inf) / norm(M0(:),Inf);
    Grad_Xk = (reshape(A'*(A*XkYkt(:)), [n,n]) - Aty) * Yk + lambda * DiffT(Diff(Xk)) + 0.5 * Xk * (Xk'*Xk-Yk'*Yk);
    X = Xk - lr*Grad_Xk;
    Grad_Yk = (reshape(A'*(A*XkYkt(:)), [n,n]) - Aty)' * Xk + lambda * DiffT(Diff(Yk)) + 0.5 * Yk * (Yk'*Yk-Xk'*Xk);
    Y = Yk - lr*Grad_Yk;
end

%% results display
close all;
set(0, 'DefaultLineLineWidth', 1); % 设置所有文字（标题、标签、图例等）为加粗
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontWeight', 'bold');  % 坐标轴文字
set(0, 'DefaultAxesTitleFontWeight', 'bold');  % 坐标轴文字

iter_show = 500;

figure(1)
subplot 131
plot(1:iter_show, Error_Spec_LRMF_SMS_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Spec_sLRMF_SMS_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Spec_sLRMF_SMS_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert/\left\Vert\mathbf{M}_\star\right\Vert$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix sensing');
grid on; grid minor;
subplot 132
plot(1:iter_show, Error_Fro_LRMF_SMS_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Fro_sLRMF_SMS_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Fro_sLRMF_SMS_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}/\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix sensing');
grid on; grid minor;
subplot 133
plot(1:iter_show, Error_Inf_LRMF_SMS_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Inf_sLRMF_SMS_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Inf_sLRMF_SMS_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_{\infty}/\left\Vert\mathbf{M}_\star\right\Vert_\infty$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix sensing');
grid on; grid minor;
set(gcf, 'Position', [600, 600, 750, 200]);

figure(2)
subplot 131
plot(1:iter_show, log(Error_Spec_LRMF_SMS_SI(1:iter_show)-Error_Spec_LRMF_SMS_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Spec_sLRMF_SMS_SI(1:iter_show)-Error_Spec_sLRMF_SMS_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Spec_sLRMF_SMS_SSI(1:iter_show)-Error_Spec_sLRMF_SMS_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\log\left(\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert-\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert\right)$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix sensing');
grid on; grid minor;
subplot 132
plot(1:iter_show, log(Error_Fro_LRMF_SMS_SI(1:iter_show)-Error_Fro_LRMF_SMS_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Fro_sLRMF_SMS_SI(1:iter_show)-Error_Fro_sLRMF_SMS_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Fro_sLRMF_SMS_SSI(1:iter_show)-Error_Fro_sLRMF_SMS_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\log\left(\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}-\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}\right)$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix sensing');
grid on; grid minor;
subplot 133
plot(1:iter_show, log(Error_Inf_LRMF_SMS_SI(1:iter_show)-Error_Inf_LRMF_SMS_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Inf_sLRMF_SMS_SI(1:iter_show)-Error_Inf_sLRMF_SMS_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Inf_sLRMF_SMS_SSI(1:iter_show)-Error_Inf_sLRMF_SMS_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration number $t$','interpreter','Latex');
ylabel('$\log\left(\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_{\infty}-\left\Vert\hat{\mathbf{X}}\hat{\mathbf{Y}}^\top-\mathbf{M}_\star\right\Vert_\infty\right)$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+','location','best');
title('matrix sensing');
grid on; grid minor;
set(gcf, 'Position', [600, 300, 750, 200]);



