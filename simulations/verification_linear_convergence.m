%% simulation verification
% the linear convergence behaviour of the gradient descent algroithm with spectral initialization
% for sLRMF (LRMF as a beseline)
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
lr = 0.001; % learning rate
M = M0 + sigma * randn(n,n);

num_iter = 300;

Error_Spec_LRMF_SI = zeros(1, num_iter);
Error_Fro_LRMF_SI = zeros(1, num_iter);
Error_Inf_LRMF_SI = zeros(1, num_iter);

Error_Spec_sLRMF_SI = zeros(1, num_iter);
Error_Fro_sLRMF_SI = zeros(1, num_iter);
Error_Inf_sLRMF_SI = zeros(1, num_iter);

Error_Spec_sLRMF_SSI = zeros(1, num_iter);
Error_Fro_sLRMF_SSI = zeros(1, num_iter);
Error_Inf_sLRMF_SSI = zeros(1, num_iter);

%% run LRMF with spectral intinalization
lambda = 0.1 * sigma * sqrt(n);

% spectral initialization
[U, S, V] = svds(M, r);
X = U * sqrt(S);
Y = V * sqrt(S);

% main
for iter = 1:num_iter
    Xk = X; Yk = Y;
    iterate = Xk * Yk';
    Error_Spec_LRMF_SI(iter) = norm(iterate - M0) / norm(M0);
    Error_Fro_LRMF_SI(iter) = norm(iterate - M0, 'fro') / norm(M0, 'fro');
    Error_Inf_LRMF_SI(iter) = norm(iterate(:) - M0(:), Inf) / norm(M0(:), Inf);
    Grad_Xk = (Xk * Yk' - M) * Yk + lambda * Xk + 0.5 * Xk * (Xk' * Xk - Yk' * Yk);
    X = Xk - lr * Grad_Xk;
    Grad_Yk = (Xk * Yk' - M)' * Xk + lambda * Yk + 0.5 * Yk * (Yk' * Yk - Xk' * Xk);
    Y = Yk - lr * Grad_Yk;
end

%% run sLRMF with spectral intinalization
lambda = 2 * sigma * sqrt(n);

% spectral initialization
[U, S, V] = svds(M, r);
X = U * sqrt(S);
Y = V * sqrt(S);

% main
for iter = 1:num_iter
    Xk = X; Yk = Y;
    iterate = Xk * Yk';
    Error_Spec_sLRMF_SI(iter) = norm(iterate - M0) / norm(M0);
    Error_Fro_sLRMF_SI(iter) = norm(iterate - M0, 'fro') / norm(M0, 'fro');
    Error_Inf_sLRMF_SI(iter) = norm(iterate(:) - M0(:), Inf) / norm(M0(:), Inf);
    Grad_Xk = (Xk * Yk' - M) * Yk + lambda * DiffT(Diff(Xk)) + 0.5 * Xk * (Xk' * Xk - Yk' * Yk);
    X = Xk - lr * Grad_Xk;
    Grad_Yk = (Xk * Yk' - M)' * Xk + lambda * DiffT(Diff(Yk)) + 0.5 * Yk * (Yk' * Yk - Xk' * Xk);
    Y = Yk - lr * Grad_Yk;
end

%% run sLRMF with smooth spectral intinalization
gamma = 0.5 * lambda / S0(r,r);
% smoothed spectral initialization
[X, Y] = smoothed_spectral_initialization(M, r, gamma);

% main
for iter = 1:num_iter
    Xk = X; Yk = Y;
    iterate = Xk * Yk';
    Error_Spec_sLRMF_SSI(iter) = norm(iterate - M0) / norm(M0);
    Error_Fro_sLRMF_SSI(iter) = norm(iterate - M0, 'fro') / norm(M0, 'fro');
    Error_Inf_sLRMF_SSI(iter) = norm(iterate(:) - M0(:), Inf) / norm(M0(:), Inf);
    Grad_Xk = (Xk * Yk' - M) * Yk + lambda * DiffT(Diff(Xk)) + 0.5 * Xk * (Xk' * Xk - Yk' * Yk);
    X = Xk - lr * Grad_Xk;
    Grad_Yk = (Xk * Yk' - M)' * Xk + lambda * DiffT(Diff(Yk)) + 0.5 * Yk * (Yk' * Yk - Xk' * Xk);
    Y = Yk - lr * Grad_Yk;
end

%% results display
set(0, 'DefaultLineLineWidth', 1);
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultAxesTitleFontWeight', 'bold');

iter_show = 300;
figure(1)
subplot 131
plot(1:iter_show, Error_Spec_LRMF_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Spec_sLRMF_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Spec_sLRMF_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert/\left\Vert\mathbf{M}_\star\right\Vert$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+');
grid on; grid minor;
subplot 132
plot(1:iter_show, Error_Fro_LRMF_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Fro_sLRMF_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Fro_sLRMF_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}/\left\Vert\mathbf{M}_\star\right\Vert_\mathrm{F}$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+');
grid on; grid minor;
subplot 133
plot(1:iter_show, Error_Inf_LRMF_SI(1:iter_show), '-b'); hold on;
plot(1:iter_show, Error_Inf_sLRMF_SI(1:iter_show), '-r'); hold on;
plot(1:iter_show, Error_Inf_sLRMF_SSI(1:iter_show), '-.r'); hold on;
xlabel('iteration $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_{\infty}/\left\Vert\mathbf{M}_\star\right\Vert_{\infty}$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+');
grid on; grid minor;
set(gcf, 'Position', [600, 600, 750, 200]);

figure(2)
subplot 131
plot(1:iter_show, log(Error_Spec_LRMF_SI(1:iter_show)-Error_Spec_LRMF_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Spec_sLRMF_SI(1:iter_show)-Error_Spec_sLRMF_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Spec_sLRMF_SSI(1:iter_show)-Error_Spec_sLRMF_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+');
grid on; grid minor;
subplot 132
plot(1:iter_show, log(Error_Fro_LRMF_SI(1:iter_show)-Error_Fro_LRMF_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Fro_sLRMF_SI(1:iter_show)-Error_Fro_sLRMF_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Fro_sLRMF_SSI(1:iter_show)-Error_Fro_sLRMF_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_\mathrm{F}$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+');
grid on; grid minor;
subplot 133
plot(1:iter_show, log(Error_Inf_LRMF_SI(1:iter_show)-Error_Inf_LRMF_SI(iter_show)), '-b'); hold on;
plot(1:iter_show, log(Error_Inf_sLRMF_SI(1:iter_show)-Error_Inf_sLRMF_SI(iter_show)), '-r'); hold on;
plot(1:iter_show, log(Error_Inf_sLRMF_SSI(1:iter_show)-Error_Inf_sLRMF_SSI(iter_show)), '-.r'); hold on;
xlabel('iteration $t$','interpreter','Latex');
ylabel('$\left\Vert\mathbf{X}_t\mathbf{Y}_t^\top-\mathbf{M}_\star\right\Vert_{\infty}$','interpreter','Latex');
legend('LRMF','sLRMF', 'sLRMF+');
grid on; grid minor;
set(gcf, 'Position', [600, 300, 750, 200]);

