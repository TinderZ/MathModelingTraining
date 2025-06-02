t_mod = (1790 + 10 * (0:22))'; 
x_mod = [3.9; 5.3; 7.2; 9.6; 12.9; 17.1; 23.2; 31.4; 38.6; 50.2; 62.9;76; 92; 105.7; 122.8; 131.7; 150.7; 179.3; 203.2; 226.5; 248.7; 281.4; 308.7]; % 带有噪声的指数增长数据

% 1. 数值微分估计 dx/dt
dx_dt_est_mod = zeros(length(x_mod), 1);
if length(t_mod) > 1
    if length(t_mod) > 2
        for i = 2:length(t_mod)-1
            dx_dt_est_mod(i) = (x_mod(i+1) - x_mod(i-1)) / (t_mod(i+1) - t_mod(i-1));
        end
    end
    if length(t_mod) >= 2
        dx_dt_est_mod(1) = (x_mod(2) - x_mod(1)) / (t_mod(2) - t_mod(1));
        dx_dt_est_mod(end) = (x_mod(end) - x_mod(end-1)) / (t_mod(end) - t_mod(end-1));
    elseif length(t_mod) == 1
        dx_dt_est_mod(1) = NaN;
    end
else
    dx_dt_est_mod(:) = NaN;
end

% 2. 计算 g(t) = (1/x) * (dx/dt)
g_t = dx_dt_est_mod ./ x_mod;

% 清理 NaN 和 Inf 值，确保 t_mod 和 g_t 对应
valid_indices = ~isnan(g_t) & ~isinf(g_t);
t_for_g_fit = t_mod(valid_indices);
g_t_for_fit = g_t(valid_indices);

if length(t_for_g_fit) < 2
    error('数值微分方法（改进模型）：没有足够的有效点来拟合 g(t)。');
end

% 3. 对 g(t) = r0 - r1*t进行线性回归
X_design_g = [ones(size(t_for_g_fit)), t_for_g_fit]; % 列对应 r0 和 -r1
params_g = X_design_g \ g_t_for_fit;

r0_hat_mod_nd = params_g(1);
minus_r1_hat_mod_nd = params_g(2);
r1_hat_mod_nd = -minus_r1_hat_mod_nd;

% 4. 估计 x0
% 使用所有有效数据点进行估计
ln_x0_estimates_mod = log(x_mod(valid_indices)) - r0_hat_mod_nd * t_mod(valid_indices) + (r1_hat_mod_nd/2) * t_mod(valid_indices).^2;
x0_hat_mod_nd = exp(mean(ln_x0_estimates_mod));


% 显示结果
fprintf('\n数值微分方法估计 (x(t) = x0*e^(r0*t - r1*t^2/2)):\n');
fprintf('x0_hat = %.4f\n', x0_hat_mod_nd);
fprintf('r0_hat = %.4f\n', r0_hat_mod_nd);
fprintf('r1_hat = %.4f\n', r1_hat_mod_nd);

% 可选：绘制原始数据和拟合曲线
figure;
plot(t_mod, x_mod, 'o', 'DisplayName', '原始数据 (改进模型)');
hold on;
t_fit_mod_nd = linspace(min(t_mod), max(t_mod), 100);
x_fit_mod_nd = x0_hat_mod_nd * exp(r0_hat_mod_nd * t_fit_mod_nd - (r1_hat_mod_nd/2) * t_fit_mod_nd.^2);
plot(t_fit_mod_nd, x_fit_mod_nd, '--', 'DisplayName', '数值微分拟合 (改进模型)');
xlabel('时间 (t)');
ylabel('人口 (x)');
legend;
title('改进的指数增长模型 - 数值微分方法');
grid on;

% 可选：绘制 g(t) 的拟合情况
figure;
plot(t_for_g_fit, g_t_for_fit, 's', 'DisplayName', 'g(t) 数值估计');
hold on;
g_t_fit_line = r0_hat_mod_nd - r1_hat_mod_nd * t_fit_mod_nd;
plot(t_fit_mod_nd, g_t_fit_line, 'r-', 'DisplayName', 'g(t) 线性拟合');
xlabel('时间 (t)');
ylabel('g(t) = (1/x) dx/dt');
legend;
title('g(t) 拟合用于改进模型参数估计');
grid on;