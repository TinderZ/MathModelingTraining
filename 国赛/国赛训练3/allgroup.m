% 输入数据

A_media = [
    0,10, 4, 50, 5, 0,  2;
    0,10, 30, 5, 12, 0, 0;
    20, 0, 0, 0, 0,5, 3;
    8, 0, 0, 0, 0,6,  10;
    0,6, 5, 10, 11, 4, 0
];
min_values = [25; 40; 60; 120; 40; 11; 15];
max_values = [60; 70; 120; 140; 80; 25; 55];

% 生成资金范围
total_cost_values = 6.4:0.0001:14;
total_cost_values = total_cost_values(total_cost_values <= 14);  
num_points = length(total_cost_values);

% 初始化存储结果
audience_total = nan(num_points, 1);        % 总观众数
audience_groups = nan(num_points, 7);       % 各群体观众数
media_investment = nan(5, num_points);      % 媒体投入

% 约束条件设置
A = []; b = [];
for j = 1:7
    row = [-A_media(:,j)', zeros(1,j-1), 1, zeros(1,7-j)];
    A = [A; row];
    b = [b; 0];
end

lb = [zeros(5,1); min_values];
ub = [inf(5,1); max_values];
Aeq = [ones(1,5), zeros(1,7)];
beq = 0;
f = [zeros(5,1); -ones(7,1)];
options = optimoptions('linprog', 'Display', 'none', 'Algorithm', 'dual-simplex');

% 进度显示设置
fprintf('开始计算（共%d个资金点）...\n', num_points);
progress_step = floor(num_points/10);

% 求解并记录数据
for k = 1:num_points
    % 更新进度显示
    if mod(k, progress_step) == 0
        fprintf('已完成%.1f%%...\n', k/num_points*100);
    end
    
    beq(1) = total_cost_values(k);
    [sol, ~, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub, options);
    
    if exitflag == 1
        audience_total(k) = sum(sol(6:12));
        audience_groups(k,:) = sol(6:12)';
        media_investment(:,k) = sol(1:5);
    end
end

% 过滤有效数据
valid_idx = ~isnan(audience_total);
plot_cost = total_cost_values(valid_idx);
plot_total = audience_total(valid_idx);
plot_groups = audience_groups(valid_idx,:);

%% 绘图

% 绘图设置
figure;
hold on;
group_names = {'群体1','群体2','群体3','群体4','群体5','群体6','群体7'};
colors = lines(8);

% 绘制各群体曲线（使用更高效的绘图方式）
group_plots = gobjects(7,1);  % 图形对象预分配
for g = 1:7
    group_plots(g) = plot(plot_cost, plot_groups(:,g),...
        'Color', colors(g,:),...
        'LineWidth', 1.2,...
        'DisplayName', group_names{g});
end

% 绘制总观众数曲线
total_plot = plot(plot_cost, plot_total,...
    'Color', [0 0 0],...
    'LineWidth', 2,...
    'LineStyle', '--',...
    'DisplayName', '总观众数');


title('观众人数分布分析（步长0.02）', 'FontSize', 12);
legend([group_plots; total_plot], 'Location', 'northwest');
grid on;

% 标注最大限制线（优化性能）
max_lines = gobjects(7,1);
for g = 1:7
    max_lines(g) = yline(max_values(g),...
        'Color', [colors(g,:) 0.3],...  % 带透明度
        'LineStyle', ':',...
        'LineWidth', 0.6,...
        'HandleVisibility','off');
end

% 设置坐标轴范围
xlim([6.5, 14]);
ylim([0 max(max_values)*1.1]);
hold off;

% 优化渲染性能
set(gcf, 'Renderer', 'painters');
