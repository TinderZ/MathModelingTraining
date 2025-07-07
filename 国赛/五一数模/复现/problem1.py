import numpy as np  # 数值计算
from pyswarm import pso  # 粒子群优化算法
import pandas as pd  # 数据处理
import matplotlib.pyplot as plt  # 数据可视化

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取 Excel 文件
file = pd.ExcelFile('E:/AAD竞赛/数模/五一建模/2025-51MCM-Problem A/附件(Attachment).xlsx')
# 获取 表1 中的数据
data = file.parse('表1 (Table 1)')
# 提取时间和车流量数据
t = data['时间 t (Time t)'].values
flow_observed = data['主路3的车流量 (Traffic flow on the Main road 3)'].values

# 定义车流量预测模型函数
def FlowModel(params, t):
    a1, b1, c1, d1, c2, d2, t_peak = params  # 解包参数

    flow1 = a1 * t + b1  # 支路1车流量线性模型

    flow2 = np.where(t <= t_peak,
                  c1 * t + d1,
                  c2 * (t - t_peak) + (c1 * t_peak + d1))  # 支路2车流量线性模型

    return flow1 + flow2

# 定义均方误差计算函数
def mse(params):
    flow_pred = FlowModel(params, t)
    a1, b1, c1, d1, c2, d2, t_peak = params
    flow1 = a1 * t + b1
    flow2 = np.where(t <= t_peak, c1 * t + d1, c2 * (t - t_peak) + (c1 * t_peak + d1))
    # 添加约束条件，若 flow1 或 flow2 小于 0，添加惩罚项
    penalty = 0
    if np.any(flow1 < 0):
        penalty += np.sum(np.abs(flow1[flow1 < 0])) * 1000
    if np.any(flow2 < 0):
        penalty += np.sum(np.abs(flow2[flow2 < 0])) * 1000
    mse = np.mean((flow_pred - flow_observed) ** 2) + penalty
    return mse

# 定义参数边界
lb = [0, 0, 0, 0, -3, 0, 28]  # 下界
ub = [2, 10, 2, 10, 0, 100, 32]  # 上界
# 将t_peak参数范围确定在[28,32]

# 设置多次运行参数
num_runs = 10  # PSO算法运行次数
all_best_params = []  # 存储每次运行得到的最佳参数
all_best_mse = []  # 存储每次运行得到的最小MSE

# 多次运行 PSO 算法提高结果稳定性
for _ in range(num_runs):
    # 使用PSO优化
    best_params, best_mse = pso(mse, lb, ub, swarmsize=100, maxiter=1000)  # swarmsize=100（粒子群大小） maxiter=1000（最大迭代次数）
    all_best_params.append(best_params)  # 保存本次运行的最佳参数
    all_best_mse.append(best_mse)  # 保存本次运行的最小MSE

# 取平均值
average_best_params = np.mean(all_best_params, axis=0)
average_best_mse = np.mean(all_best_mse)

# 输出结果
print("最优参数:")
print(f"a1 = {average_best_params[0]:.4f}, b1 = {average_best_params[1]:.4f}")
print(f"c1 = {average_best_params[2]:.4f}, d1 = {average_best_params[3]:.4f}")
print(f"c2 = {average_best_params[4]:.4f}, d2 = {average_best_params[5]:.4f}")
print(f"t_peak = {average_best_params[6]:.1f}")
print(f"最小均方误差(MSE): {average_best_mse:.10f}")

# 绘制结果
flow_pred = FlowModel(average_best_params, t)
flow1 = average_best_params[0] * t + average_best_params[1]
flow2 = np.where(t <= average_best_params[6],
              average_best_params[2] * t + average_best_params[3],
              average_best_params[4] * (t - average_best_params[6]) + (
                          average_best_params[2] * average_best_params[6] + average_best_params[3]))

plt.figure(figsize=(12, 6))
plt.plot(t, flow_observed, 'bo-', label='主路3实际车流量')
plt.plot(t, flow_pred, 'r--', label='主路3预测车流量')
plt.plot(t, flow1, 'g-', label='支路1车流量')
plt.plot(t, flow2, 'm-', label='支路2车流量')
plt.xlabel('时间t (相对于7:00的分钟数/2)')
plt.ylabel('车流量')
plt.title('问题1：主路3和各支路车流量')
plt.legend()
plt.grid(True)
plt.show()

# 绘制支路车流量叠加图
plt.figure(figsize=(12, 6))
plt.plot(t, flow_observed, 'bo-', label='主路3实际车流量', alpha=0.5)
plt.plot(t, flow_pred, 'r--', label='主路3预测车流量')
plt.fill_between(t, 0, flow1, alpha=0.3, label='支路1车流量')
plt.fill_between(t, flow1, flow1 + flow2, alpha=0.3, label='支路2车流量')
plt.xlabel('时间t (相对于7:00的分钟数/2)')
plt.ylabel('车流量')
plt.title('问题1：支路车流量叠加及主路车流量比较')
plt.grid(True)
plt.legend()

# 输出函数表达式
print("\n表1.1 问题1支路车流量函数表达式")
print("--------------------------------------------------")
print(f"支路1: flow1(t) = {average_best_params[0]:.4f}t + {average_best_params[1]:.4f}")
print(f"支路2: flow2(t) = {average_best_params[2]:.4f}t + {average_best_params[3]:.4f} (t ≤ {average_best_params[6]:.1f})")
print(
    f"       flow2(t) = {average_best_params[4]:.4f}(t-{average_best_params[6]:.1f}) + {average_best_params[2] * average_best_params[6] + average_best_params[3]:.4f} (t > {average_best_params[6]:.1f})")
print("--------------------------------------------------")

# 灵敏度分析
param_names = ['a1', 'b1', 'c1', 'd1', 'c2', 'd2', 't_peak']
sensitivity = []
perturbation = 0.01  # 扰动比例

for i in range(len(average_best_params)):
    params_perturbed = average_best_params.copy()
    params_perturbed[i] *= (1 + perturbation)
    mse_perturbed = mse(params_perturbed)
    sensitivity.append((mse_perturbed - average_best_mse) / (average_best_params[i] * perturbation))

# 输出灵敏度分析结果
print("\n灵敏度分析结果:")
for i in range(len(param_names)):
    print(f"{param_names[i]} 的灵敏度: {sensitivity[i]:.4f}")

# 绘制灵敏度分析结果
plt.figure(figsize=(10, 6))
color = (178/255, 221/255, 202/255)
plt.bar(param_names, sensitivity, color=color)
plt.xlabel('参数')
plt.ylabel('灵敏度')
plt.title('问题1：参数灵敏度分析')
plt.grid(True)
plt.show()