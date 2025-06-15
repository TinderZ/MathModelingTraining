import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, dual_annealing
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from statsmodels.robust import mad

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取表4数据
data = pd.read_excel('E:/AAD竞赛/数模/五一建模/2025-51MCM-Problem A/附件(Attachment).xlsx', sheet_name='表4 (Table 4)')
t = data['时间 t (Time t)'].values
flow = data['主路4的车流量 (Traffic flow on the Main road 4)'].values

# 根据题目描述，车辆从支路1和支路2的路口处到达A3处的行驶时间为2分钟
delay = 1  # 对应2分钟

# 鲁棒主成分分析清洗数据
def robust_pca_denoise(data, n_components=1, threshold=3):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data.reshape(-1, 1))
    reconstructed = pca.inverse_transform(reduced).flatten()
    residuals = data - reconstructed
    median = np.median(residuals)
    mad_score = mad(residuals)

    # 检查 mad_score 是否为零
    if mad_score == 0:
        z_scores = np.zeros_like(residuals)
    else:
        z_scores = (residuals - median) / mad_score

    cleaned = np.where(np.abs(z_scores) > threshold, reconstructed + median, data)
    return cleaned

flow_cleaned = robust_pca_denoise(flow)

# 信号灯C的设置
red_time = 4  # 红灯时间8分钟，对应4个时间单位
green_time = 5  # 绿灯时间10分钟，对应5个时间单位
cycle_time = red_time + green_time  # 周期时间

# 滑动窗口法检测信号灯周期
def detect_traffic_light(flow, t, window_size=5, threshold=0.7):
    # 计算滑动平均
    rolling_mean = pd.Series(flow).rolling(window=window_size, center=True).mean().values
    # 计算变化率
    diff = np.diff(rolling_mean, prepend=0)
    # 标准化变化率
    norm_diff = (diff - np.mean(diff)) / np.std(diff)

    # 寻找显著上升沿作为绿灯开始
    green_starts = []
    in_green = False
    for i in range(1, len(norm_diff)):
        if not in_green and norm_diff[i] > threshold and norm_diff[i] > norm_diff[i - 1]:
            green_starts.append(t[i])
            in_green = True
        elif in_green and norm_diff[i] < -threshold and norm_diff[i] < norm_diff[i - 1]:
            in_green = False

    # 计算平均周期
    if len(green_starts) > 1:
        periods = np.diff(green_starts)
        avg_period = np.mean(periods)
        first_green_start = green_starts[0]
    else:
        avg_period = cycle_time
        first_green_start = t[np.argmax(flow > np.mean(flow))]

    return first_green_start, avg_period

first_green_start, avg_period = detect_traffic_light(flow_cleaned, t)

# 生成新的绿灯开始时间序列
def generate_green_starts(first_green_start, avg_period, t_max):
    # 将找到的第一个绿灯时间作为第二个绿灯时间
    second_green_start = first_green_start
    # 往前取一个绿灯周期
    first_green_start = second_green_start - avg_period
    # 生成绿灯开始时间序列
    num_cycles = int((t_max - first_green_start) // avg_period) + 1
    green_starts = [first_green_start + i * avg_period for i in range(num_cycles)]
    return green_starts

# 生成新的绿灯开始时间序列
green_starts = generate_green_starts(first_green_start, avg_period, t.max())

# 判断某时刻是否为绿灯
def is_green(t, green_starts):
    for start in green_starts:
        elapsed_time = t - start  # 计算相对于当前绿灯开始的时间差
        if elapsed_time < 0:
            adjusted_time = elapsed_time % cycle_time  # 对负时间取模
            return adjusted_time >= -green_time
        else:
            cycle_position = elapsed_time % cycle_time  # 计算此时间在当前周期中的位置
            return cycle_position < green_time
    return False

# 计算某个时间范围内的信号灯状态
def get_traffic_light_status(t_range):
    return np.array([is_green(t_val, green_starts) for t_val in t_range])

# 定义支路车流量的参数
def branch_flow(t, params):
    a1, a2, a3, t_break1, t_break2, t_break3 = params[:6]
    b1, b2, b3, b4, b5 = params[6:11]
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = params[11:]

    # 支路1：分段函数，无车流量→线性增长→稳定→线性减少至无车流量
    flow1 = np.zeros_like(t, dtype=float)
    flow1[t < t_break1] = 0
    mask_growth = (t >= t_break1) & (t < t_break2)
    flow1[mask_growth] = a1 * (t[mask_growth] - t_break1) + a2
    mask_stable = (t >= t_break2) & (t < t_break3)
    flow1[mask_stable] = a1 * (t_break2 - t_break1) + a2
    mask_decrease = (t >= t_break3)
    flow1[mask_decrease] = a3 * (t[mask_decrease] - t_break3) + (a1 * (t_break2 - t_break1) + a2)
    flow1 = np.maximum(flow1, 0)

    # 支路2：分段线性函数
    flow2 = np.zeros_like(t, dtype=float)
    mask_growth1 = (t <= 17)
    flow2[mask_growth1] = b1 * t[mask_growth1] + b2
    mask_stable1 = (t > 17) & (t <= 35)
    flow2[mask_stable1] = b3
    mask_decrease1 = (t > 35)
    flow2[mask_decrease1] = b4 * (t[mask_decrease1] - 35) + b5
    flow2 = np.maximum(flow2, 0)

    # 支路3：受信号灯控制
    traffic_light = get_traffic_light_status(t)  # 获取信号灯状态
    flow3 = np.zeros_like(t, dtype=float)
    # 动态生成所有绿灯周期
    num_cycles = int((t.max() - green_starts[0]) // cycle_time) + 1  # 计算完整的信号周期数
    cycle_starts = [green_starts[0] + i * cycle_time for i in range(num_cycles)]  # 生成每个周期的开始时间
    params_per_cycle = [(c1, c2, c3, c4), (c5, c6, c7, c8), (c9, c10, c11, c12)] * (num_cycles // 3 + 1)  # 为每个周期分配参数
    # 为每个绿灯周期计算车流量
    for cycle_idx, cycle_start in enumerate(cycle_starts):
        mask_green = (t >= cycle_start) & (t < cycle_start + green_time) & traffic_light
        slope1, intercept1, slope2, intercept2 = params_per_cycle[cycle_idx % len(params_per_cycle)]
        # 线性增长阶段
        growth_end = cycle_start + int(green_time / 3)
        mask_growth2 = (t >= cycle_start) & (t < growth_end) & mask_green
        flow3[mask_growth2] = slope1 * (t[mask_growth2] - cycle_start) + intercept1
        # 稳定阶段
        stable_end = cycle_start + 2 * int(green_time / 3)
        mask_stable2 = (t >= growth_end) & (t < stable_end) & mask_green
        stable_value = slope1 * (growth_end - cycle_start) + intercept1
        flow3[mask_stable2] = stable_value
        # 线性减少阶段
        mask_decrease2 = (t >= stable_end) & (t < cycle_start + green_time) & mask_green
        flow3[mask_decrease2] = slope2 * (t[mask_decrease2] - stable_end) + stable_value

    flow3 = np.maximum(flow3, 0)  # 确保车流量非负

    return flow1, flow2, flow3

# 考虑延迟，计算主路4上的车流量
def main_flow(t, params):
    flow1, flow2, flow3 = branch_flow(t, params)
    # 初始化延迟后的流量数组
    flow1_delayed = np.zeros_like(t, dtype=float)
    flow2_delayed = np.zeros_like(t, dtype=float)
    # 处理延迟
    mask = t >= delay
    flow1_delayed[mask] = flow1[np.where(mask)[0] - delay]
    flow2_delayed[mask] = flow2[np.where(mask)[0] - delay]
    flow1_delayed[~mask] = flow1[0]
    flow2_delayed[~mask] = flow2[0]
    return flow1_delayed + flow2_delayed + flow3

# 定义目标函数：计算预测值与实际值的均方误差
def objective(params):
    flow_pred = main_flow(t, params)
    mse = np.mean((flow_pred - flow) ** 2)
    flow1, flow2, flow3 = branch_flow(t, params)
    penalty = 0
    # 为负流量添加惩罚
    if np.any(flow1 < 0):
        penalty += 1000 * np.sum(np.abs(flow1[flow1 < 0]))
    if np.any(flow2 < 0):
        penalty += 1000 * np.sum(np.abs(flow2[flow2 < 0]))
    if np.any(flow3 < 0):
        penalty += 1000 * np.sum(np.abs(flow3[flow3 < 0]))

    # 添加连续性约束
    a1, a2, a3, t_break1, t_break2, t_break3 = params[:6]
    b1, b2, b3, b4, b5 = params[6:11]

    # 支路1连续性约束
    continuity_penalty1 = 0
    # 从0到线性增长
    start_growth_value = a1 * (t_break1 - t_break1) + a2
    continuity_penalty1 += 1000 * (start_growth_value - 0)**2
    # 线性增长到稳定
    growth_end_value = a1 * (t_break2 - t_break1) + a2
    stable_value = a1 * (t_break2 - t_break1) + a2
    continuity_penalty1 += 1000 * (growth_end_value - stable_value)**2
    # 稳定到线性减少
    stable_end_value = a1 * (t_break2 - t_break1) + a2
    decrease_start_value = a3 * (t_break3 - t_break3) + (a1 * (t_break2 - t_break1) + a2)
    continuity_penalty1 += 1000 * (stable_end_value - decrease_start_value)**2

    # 支路2连续性约束
    continuity_penalty2 = 0
    # 线性增长到稳定
    growth_end_value2 = b1 * 17 + b2
    stable_value2 = b3
    continuity_penalty2 += 1000 * (growth_end_value2 - stable_value2)**2
    # 稳定到线性减少
    stable_end_value2 = b3
    decrease_start_value = b4 * (35 - 35) + b5
    continuity_penalty2 += 1000 * (stable_end_value2 - decrease_start_value)**2

    penalty += continuity_penalty1 + continuity_penalty2

    # 添加约束条件：t = 59 时，flow1 = 0
    flow1_at_59 = branch_flow(np.array([59]), params)[0][0]
    penalty += 1000 * (flow1_at_59 - 0) ** 2

    return mse + penalty

# 设定初始参数
initial_params = [
    0.1, 0.0, -0.1, 4.0, 10.0, 20.0,  # 支路1参数
    0.5, 0.0, 20.0, -1.0, 20.0,        # 支路2参数
    5.0, 0.0, 5.0, -5.0,               # 支路3参数（第1周期）
    5.0, 0.0, 5.0, -5.0,               # 支路3参数（第2周期）
    5.0, 0.0, 5.0, -5.0                # 支路3参数（第3周期）
]

# 设置参数边界
bounds = [
    (0.0, 1.0), (0.0, 10.0), (-1.0, 0.0), (3.0, 5.0), (8.0, 12.0), (18.0, 22.0),  # 支路1边界
    (0.1, 2.0), (0.0, 10.0), (15.0, 30.0), (-2.0, -0.1), (15.0, 30.0),           # 支路2边界
    (0.0, 10.0), (0.0, 40.0), (-10.0, 0.0), (0.0, 40.0),                          # 支路3边界（第1周期）
    (0.0, 10.0), (0.0, 40.0), (-10.0, 0.0), (0.0, 40.0),                          # 支路3边界（第2周期）
    (0.0, 10.0), (0.0, 40.0), (-10.0, 0.0), (0.0, 40.0)                           # 支路3边界（第3周期）
]

# 使用模拟退火算法进行优化
result = dual_annealing(objective, bounds, x0=initial_params)
params = result.x  # 获取优化后的参数

# 提取参数
a1, a2, a3, t_break1, t_break2, t_break3 = params[:6]
b1, b2, b3, b4, b5 = params[6:11]
c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = params[11:]

# 计算各支路的流量
flow1, flow2, flow3 = branch_flow(t, params)

# 计算主路4的预测流量
flow_pred = main_flow(t, params)

# 计算均方根误差(RMSE)
rmse = np.sqrt(np.mean((flow_pred - flow) ** 2))

# 计算平均绝对误差（MAE）
mae = np.mean(np.abs(flow_pred - flow))

# 计算决定系数（R^2）
ss_res = np.sum((flow - flow_pred) ** 2)
ss_tot = np.sum((flow - np.mean(flow)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# 计算7:30和8:30时刻的各支路车流量
t_730 = 15
t_830 = 45
flow1_730, flow2_730, flow3_730 = branch_flow(np.array([t_730]), params)
flow1_830, flow2_830, flow3_830 = branch_flow(np.array([t_830]), params)

# 绘制主路预测结果
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(t, flow, 'bo-', label='主路4实际车流量')
plt.plot(t, flow_pred, 'r--', label='主路4预测车流量')
plt.xlabel('时间t (相对于7:00的分钟数/2)')
plt.ylabel('车流量')
plt.title('问题4：主路4车流量预测结果')
plt.grid(True)
plt.legend()

# 绘制各支路流量
plt.subplot(2, 1, 2)
plt.plot(t, flow1, 'g-', label='支路1车流量')
plt.plot(t, flow2, 'm-', label='支路2车流量')
plt.plot(t, flow3, 'c-', label='支路3车流量')
plt.axvline(x=t_break1, color='g', linestyle='--', alpha=0.5, label='支路1转折点')
plt.axvline(x=t_break2, color='g', linestyle='--', alpha=0.5)
plt.axvline(x=t_break3, color='g', linestyle='--', alpha=0.5)
plt.axvline(x=17, color='m', linestyle='--', alpha=0.5, label='支路2转折点')
plt.axvline(x=35, color='m', linestyle='--', alpha=0.5)
for i in range(int((t.max() - green_starts[0]) // cycle_time + 1)):
    cycle_start = green_starts[0] + i * cycle_time
    plt.axvspan(cycle_start, cycle_start + green_time, color='green', alpha=0.1)
    if i == 0:
        start = 0
    else:
        start = green_starts[0] + (i - 1) * cycle_time + green_time
    end = cycle_start
    if start < end:
        plt.axvspan(start, end, color='red', alpha=0.1)
if green_starts[0] + int((t.max() - green_starts[0]) // cycle_time) * cycle_time + green_time < t.max():
    plt.axvspan(green_starts[0] + int((t.max() - green_starts[0]) // cycle_time) * cycle_time + green_time, t.max(), color='red', alpha=0.1)
plt.xlabel('时间t (相对于7:00的分钟数/2)')
plt.ylabel('车流量')
plt.title('问题4：各支路车流量')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Q-4_主路和支路车流量.png', dpi=300, bbox_inches='tight')

# 绘制支路车流量叠加图
plt.figure(figsize=(14, 8))
plt.plot(t, flow, 'bo-', label='主路4实际车流量', alpha=0.5)
plt.plot(t, flow_pred, 'r--', label='主路4预测车流量')
flow1_delayed = np.zeros_like(t, dtype=float)
flow2_delayed = np.zeros_like(t, dtype=float)
mask = t >= delay
flow1_delayed[mask] = flow1[np.where(mask)[0] - delay]
flow2_delayed[mask] = flow2[np.where(mask)[0] - delay]
flow1_delayed[~mask] = flow1[0]
flow2_delayed[~mask] = flow2[0]
for i in range(int((t.max() - green_starts[0]) // cycle_time + 1)):
    cycle_start = green_starts[0] + i * cycle_time
    plt.axvspan(cycle_start, cycle_start + green_time, color='green', alpha=0.1)
    if i == 0:
        start = 0
    else:
        start = green_starts[0] + (i - 1) * cycle_time + green_time
    end = cycle_start
    if start < end:
        plt.axvspan(start, end, color='red', alpha=0.1)
if green_starts[0] + int((t.max() - green_starts[0]) // cycle_time) * cycle_time + green_time < t.max():
    plt.axvspan(green_starts[0] + int((t.max() - green_starts[0]) // cycle_time) * cycle_time + green_time, t.max(), color='red', alpha=0.1)
plt.fill_between(t, 0, flow1_delayed, alpha=0.3, label='支路1车流量(延迟后)')
plt.fill_between(t, flow1_delayed, flow1_delayed + flow2_delayed, alpha=0.3, label='支路2车流量(延迟后)')
plt.fill_between(t, flow1_delayed + flow2_delayed, flow1_delayed + flow2_delayed + flow3, alpha=0.3, label='支路3车流量')
plt.xlabel('时间t (相对于7:00的分钟数/2)')
plt.ylabel('车流量')
plt.title('问题4：支路车流量叠加及主路车流量比较')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Q-4_支路车流量叠加.png', dpi=300, bbox_inches='tight')

# 灵敏度分析
sensitivity = []
params_names = ['a1', 'b1', 'a2', 't1', 't2', 't3',
'a3', 'b3', 'c2', 'a4', 'b4',
'a5.1', 'b5.1', 'a6.1', 'b6.1', 'a5.2', 'b5.2', 'a6.2', 'b6.2', 'a5.3', 'b5.3', 'a6.3', 'b6.3']

perturbation = 0.01
for i in range(len(params)):
    new_params = params.copy()
    new_params[i] *= (1 + perturbation)
    new_objective = objective(new_params)
    sensitivity.append((new_objective - objective(params)) / (perturbation * params[i]))

# 绘制灵敏度分析柱状图
plt.figure(figsize=(14, 8))
plt.bar(params_names, sensitivity, color=(178/255, 221/255, 202/255))
plt.xlabel('参数')
plt.ylabel('灵敏度')
plt.title('问题4：灵敏度分析')
plt.grid(True)
plt.tight_layout()
plt.savefig('Q-4_灵敏度分析.png', dpi=300, bbox_inches='tight')

# 保存结果到文件
with open('Q-4_result.txt', 'w', encoding='utf-8') as f:
    f.write("优化结果：\n")
    f.write(f"支路1参数 (a1, a2, a3): ({a1:.4f}, {a2:.4f}, {a3:.4f})\n")
    f.write(f"支路1转折点 (t_break1, t_break2, t_break3): ({t_break1:.4f}, {t_break2:.4f}, {t_break3:.4f})\n")
    f.write(f"支路2参数 (b1, b2, b3, b4, b5): ({b1:.4f}, {b2:.4f}, {b3:.4f}, {b4:.4f}, {b5:.4f})\n")
    f.write(f"支路3参数 (c1-c12): ({c1:.4f}, {c2:.4f}, {c3:.4f}, {c4:.4f}, {c5:.4f}, {c6:.4f}, {c7:.4f}, {c8:.4f}, {c9:.4f}, {c10:.4f}, {c11:.4f}, {c12:.4f})\n")
    f.write(f"\nRMSE: {rmse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"R^2: {r2:.6f}\n")
    f.write("\n7:30时刻各支路车流量：\n")
    f.write(f"支路1: {flow1_730[0]:.2f}\n")
    f.write(f"支路2: {flow2_730[0]:.2f}\n")
    f.write(f"支路3: {flow3_730[0]:.2f}\n")
    f.write("\n8:30时刻各支路车流量：\n")
    f.write(f"支路1: {flow1_830[0]:.2f}\n")
    f.write(f"支路2: {flow2_830[0]:.2f}\n")
    f.write(f"支路3: {flow3_830[0]:.2f}\n")
    f.write("\n函数表达式：\n")
    f.write("支路1:\n")
    f.write(f"  当 t < {t_break1:.2f} 时: f1(t) = 0\n")
    f.write(f"  当 {t_break1:.2f} <= t < {t_break2:.2f} 时: f1(t) = {a1:.4f}*(t-{t_break1:.2f}) + {a2:.4f}\n")
    f.write(f"  当 {t_break2:.2f} <= t < {t_break3:.2f} 时: f1(t) = {a1:.4f}*({t_break2:.2f}-{t_break1:.2f}) + {a2:.4f}\n")
    f.write(f"  当 t >= {t_break3:.2f} 时: f1(t) = {a3:.4f}*(t-{t_break3:.2f}) + ({a1:.4f}*({t_break2:.2f}-{t_break1:.2f}) + {a2:.4f})\n")
    f.write("\n支路2:\n")
    f.write(f"  当 t <= 17 时: f2(t) = {b1:.4f}*t + {b2:.4f}\n")
    f.write(f"  当 17 < t <= 35 时: f2(t) = {b3:.4f}\n")
    f.write(f"  当 t > 35 时: f2(t) = {b4:.4f}*(t-35) + {b5:.4f}\n")
    f.write("\n支路3:\n")
    f.write("  当信号灯为红灯时: f3(t) = 0\n")
    f.write("  当信号灯为绿灯时，分段函数如下:\n")
    params_per_cycle = [(c1, c2, c3, c4), (c5, c6, c7, c8), (c9, c10, c11, c12)]
    for i in range((t.max() - green_starts[0]) // cycle_time + 1):
        cycle_start = green_starts[0] + i * cycle_time
        slope1, intercept1, slope2, intercept2 = params_per_cycle[i % len(params_per_cycle)]
        growth_end = cycle_start + int(green_time / 3)
        stable_end = cycle_start + 2 * int(green_time / 3)
        stable_value = slope1 * (growth_end - cycle_start) + intercept1
        f.write(f"    第{i + 1}个绿灯周期 ({cycle_start}<=t<{cycle_start + green_time}): \n")
        f.write(f"      当 {cycle_start}<=t<{growth_end} 时: f3(t) = {slope1:.4f}*(t-{cycle_start}) + {intercept1:.4f}\n")
        f.write(f"      当 {growth_end}<=t<{stable_end} 时: f3(t) = {stable_value:.4f}\n")
        f.write(f"      当 {stable_end}<=t<{cycle_start + green_time} 时: f3(t) = {slope2:.4f}*(t-{stable_end}) + {stable_value:.4f}\n")
    f.write("\n灵敏度分析结果：\n")
    for i, sens in enumerate(sensitivity):
        f.write(f"参数 {i}: {sens:.6f}\n")

# 打印结果
print("优化结果：")
print(f"支路1参数 (a1, a2, a3): ({a1:.4f}, {a2:.4f}, {a3:.4f})")
print(f"支路1转折点 (t_break1, t_break2, t_break3): ({t_break1:.4f}, {t_break2:.4f}, {t_break3:.4f})")
print(f"支路2参数 (b1, b2, b3, b4, b5): ({b1:.4f}, {b2:.4f}, {b3:.4f}, {b4:.4f}, {b5:.4f})")
print(f"支路3参数 (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12): "
      f"({c1:.4f}, {c2:.4f}, {c3:.4f}, {c4:.4f}, {c5:.4f}, {c6:.4f}, {c7:.4f}, {c8:.4f}, {c9:.4f}, {c10:.4f}, {c11:.4f}, {c12:.4f})")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R^2: {r2:.6f}")
print("\n7:30时刻各支路车流量：")
print(f"支路1: {flow1_730[0]:.2f}")
print(f"支路2: {flow2_730[0]:.2f}")
print(f"支路3: {flow3_730[0]:.2f}")
print("\n8:30时刻各支路车流量：")
print(f"支路1: {flow1_830[0]:.2f}")
print(f"支路2: {flow2_830[0]:.2f}")
print(f"支路3: {flow3_830[0]:.2f}")
print("\n表4.2 问题3支路车流量数值")
print("时刻    支路1    支路2    支路3")
print(f"7:30    {flow1_730[0]:.2f}    {flow2_730[0]:.2f}    {flow3_730[0]:.2f}")
print(f"8:30    {flow1_830[0]:.2f}    {flow2_830[0]:.2f}    {flow3_830[0]:.2f}")
print("\n函数表达式：")
print("支路1:")
print(f"  当 t < {t_break1:.2f} 时: f1(t) = 0")
print(f"  当 {t_break1:.2f} <= t < {t_break2:.2f} 时: f1(t) = {a1:.4f}*(t-{t_break1:.2f}) + {a2:.4f}")
print(f"  当 {t_break2:.2f} <= t < {t_break3:.2f} 时: f1(t) = {a1:.4f}*({t_break2:.2f}-{t_break1:.2f}) + {a2:.4f}")
print(f"  当 t >= {t_break3:.2f} 时: f1(t) = {a3:.4f}*(t-{t_break3:.2f}) + ({a1:.4f}*({t_break2:.2f}-{t_break1:.2f}) + {a2:.4f})")
print("支路2:")
print(f"  当 t <= 17 时: f2(t) = {b1:.4f}*t + {b2:.4f}")
print(f"  当 17 < t <= 35 时: f2(t) = {b3:.4f}")
print(f"  当 t > 35 时: f2(t) = {b4:.4f}*(t-35) + {b5:.4f}")
print("支路3:")
print("  当信号灯为红灯时: f3(t) = 0")
print("  当信号灯为绿灯时，分段函数如下:")
params_per_cycle = [(c1, c2, c3, c4), (c5, c6, c7, c8), (c9, c10, c11, c12)]
for i in range((t.max() - green_starts[0]) // cycle_time + 1):
    cycle_start = green_starts[0] + i * cycle_time
    slope1, intercept1, slope2, intercept2 = params_per_cycle[i % len(params_per_cycle)]
    growth_end = cycle_start + int(green_time / 3)
    stable_end = cycle_start + 2 * int(green_time / 3)
    stable_value = slope1 * (growth_end - cycle_start) + intercept1
    print(f"    第{i + 1}个绿灯周期 ({cycle_start}<=t<{cycle_start + green_time}): ")
    print(f"      当 {cycle_start}<=t<{growth_end} 时: f3(t) = {slope1:.4f}*(t-{cycle_start}) + {intercept1:.4f}")
    print(f"      当 {growth_end}<=t<{stable_end} 时: f3(t) = {stable_value:.4f}")
    print(f"      当 {stable_end}<=t<{cycle_start + green_time} 时: f3(t) = {slope2:.4f}*(t-{stable_end}) + {stable_value:.4f}")

# 显示图表
plt.show()