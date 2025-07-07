import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import matplotlib.pyplot as plt  # 数据可视化
import random  # 随机数生成
import multiprocessing  # 多进程处理
from datetime import datetime  # 时间处理
import os  # 操作系统接口

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class TrafficFlowOptimizer:
    def __init__(self, data_path, delay=1):
        """初始化交通流量优化器
              参数:
                  data_path: 数据文件路径
                  delay: 支路车流到达主路的延迟时间(单位:时间步长)
              """
        self.delay = delay
        self.t = None  # 时间序列
        self.flow = None  # 主路车流量数据
        self.best_params = None  # 最佳参数
        self.best_energy = float('inf')  # 最小误差
        self.history = []  # 优化过程记录

        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            print(f"错误：数据文件路径 {data_path} 不存在，请检查路径。")
            raise SystemExit(1)
        self.load_data(data_path)  # 加载数据

    def load_data(self, data_path):
        # 从Excel文件加载数据
        try:
            # 读取表2数据
            self.data = pd.read_excel("E:/AAD竞赛/数模/五一建模/2025-51MCM-Problem A/附件(Attachment).xlsx", sheet_name="表2 (Table 2)")
            # 检查必要列是否存在
            required_columns = ['时间 t (Time t)', '主路5的车流量 (Traffic flow on the Main road 5)']
            for col in required_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Excel文件中缺少必要的列 {col}，请检查文件格式。")

            # 提取时间和车流量数据
            self.t = self.data['时间 t (Time t)'].values
            self.flow = self.data['主路5的车流量 (Traffic flow on the Main road 5)'].values
            print(f"成功从{data_path}加载数据，共{len(self.t)}条记录。")
        except Exception as e:
            print(f"加载数据时出错: {e}")
            print("请检查数据文件路径和格式，程序终止。")
            raise SystemExit(1)

    def branch_flow(self, t, params):
        """计算各支路的车流量
               参数:
                   t: 时间序列
                   params: 包含14个参数的列表
               返回:
                   四个支路的车流量数组
               """
        a, b1, b2, b3, t_break1, t_break2, c1, c2, c3, t_break3, d1, d2, d3, d4 = params  # 解包参数

        # 支路1：稳定
        flow1 = np.full_like(t, a, dtype=float)

        # 支路2：分段线性
        flow2 = np.zeros_like(t, dtype=float)
        flow2[(t <= t_break1)] = b1 * t[(t <= t_break1)] + b2
        flow2[(t > t_break1) & (t <= t_break2)] = b3
        flow2[(t > t_break2)] = b1 * (t[(t > t_break2)] - t_break2) + b3

        # 支路3：先线性增长后稳定
        flow3 = np.zeros_like(t, dtype=float)
        flow3[t <= t_break3] = c1 * t[t <= t_break3] + c2
        flow3[t > t_break3] = c3

        # 支路4：方波函数
        flow4 = d1 * np.sign(np.sin(d2 * t + d3)) + d4

        return flow1, flow2, flow3, flow4

    def main_flow(self, t, params):
        """计算主路车流量(各支路车流量的叠加)
             参数:
                 t: 时间序列
                 params: 参数列表
             返回:
                 主路车流量预测值
             """
        flow1, flow2, flow3, flow4 = self.branch_flow(t, params)

        # 延迟处理：支路1和支路2的车流需要2分钟才能到达主路5的监测点
        flow1_delayed = np.zeros_like(t, dtype=float)
        flow2_delayed = np.zeros_like(t, dtype=float)

        # 对于t>=delay的时刻，使用t-delay时刻的支路1和支路2流量
        mask = t >= self.delay
        flow1_delayed[mask] = flow1[np.where(mask)[0] - self.delay]
        flow2_delayed[mask] = flow2[np.where(mask)[0] - self.delay]

        # 对于t<delay的时刻，假设支路1和支路2的流量为初始值
        flow1_delayed[~mask] = flow1[0]
        flow2_delayed[~mask] = flow2[0]

        # 主路5车流量 = 支路1(延迟) + 支路2(延迟) + 支路3 + 支路4
        return flow1_delayed + flow2_delayed + flow3 + flow4

    def objective(self, params):
        """目标函数:计算预测值与实际值的均方误差
           参数:
               params: 参数列表
           返回:
               带惩罚项的均方误差
           """
        flow_pred = self.main_flow(self.t, params)  # 计算预测值
        mse = np.mean((flow_pred - self.flow) ** 2)  # 计算均方误差

        # 计算各支路流量
        flow1, flow2, flow3, flow4 = self.branch_flow(self.t, params)

        # 添加约束惩罚：确保所有流量非负
        penalty = 0
        if np.any(flow1 < 0):
            penalty += 1000 * np.sum(np.abs(flow1[flow1 < 0]))
        if np.any(flow2 < 0):
            penalty += 1000 * np.sum(np.abs(flow2[flow2 < 0]))
        if np.any(flow3 < 0):
            penalty += 1000 * np.sum(np.abs(flow3[flow3 < 0]))
        if np.any(flow4 < 0):
            penalty += 1000 * np.sum(np.abs(flow4[flow4 < 0]))

        # 添加平滑惩罚：防止参数变化过快
        smooth_penalty = 0
        # 对转折点参数添加平滑惩罚
        smooth_penalty += 10 * max(0, params[4] - params[5])  # 确保t_break1 <= t_break2

        # 添加连续函数约束条件
        a, b1, b2, b3, t_break1, t_break2, c1, c2, c3, t_break3, d1, d2, d3, d4 = params
        # 支路2在转折点处的连续性
        continuity_penalty = 0
        f2_left = b1 * t_break1 + b2
        f2_right = b3
        continuity_penalty += 1000 * np.abs(f2_left - f2_right)
        f2_left_2 = b3
        f2_right_2 = b1 * (t_break2 - t_break2) + b3
        continuity_penalty += 1000 * np.abs(f2_left_2 - f2_right_2)

        # 支路3在转折点处的连续性
        f3_left = c1 * t_break3 + c2
        f3_right = c3
        continuity_penalty += 1000 * np.abs(f3_left - f3_right)

        return mse + penalty + smooth_penalty + continuity_penalty

    def calculate_mae(self, pred_flow):
        # 计算平均绝对误差（MAE）
        return np.mean(np.abs(pred_flow - self.flow))

    def adaptive_simulated_annealing(self, initial_params, bounds, max_iter=1000, initial_temp=100, cooling_rate=0.95,
                                     stagnation_threshold=100, num_processes=None):
        """自适应模拟退火算法
              参数:
                  initial_params: 初始参数
                  bounds: 参数边界
                  max_iter: 最大迭代次数
                  initial_temp: 初始温度
                  cooling_rate: 冷却速率
                  stagnation_threshold: 早停阈值
                  num_processes: 使用的进程数
              返回:
                  最佳参数, 最小误差, 优化历史
              """
        # 初始化当前解和最佳解
        current_solution = np.copy(initial_params)
        best_solution = np.copy(initial_params)
        current_energy = self.objective(current_solution)
        best_energy = current_energy

        # 温度初始化
        temperature = initial_temp

        # 记录迭代过程
        self.history = []

        # 自适应参数
        step_sizes = [(upper - lower) / 10 for lower, upper in bounds]
        stagnation_count = 0
        best_energy_history = []

        # 确定使用的进程数
        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() - 1)

        print(f"使用{num_processes}个进程进行优化")

        # 多进程池
        with multiprocessing.Pool(processes=num_processes) as pool:
            for iteration in range(max_iter):
                # 生成多个邻域解
                neighbor_params = []
                for _ in range(num_processes):
                    neighbor = np.copy(current_solution)

                    # 随机选择参数进行扰动
                    param_idx = random.randint(0, len(initial_params) - 1)
                    # 跳过 t_break1 和 t_break2
                    while param_idx in [4, 5]:
                        param_idx = random.randint(0, len(initial_params) - 1)

                    # 自适应扰动步长
                    perturbation = random.uniform(-step_sizes[param_idx], step_sizes[param_idx])
                    neighbor[param_idx] += perturbation

                    # 确保参数在边界内
                    neighbor[param_idx] = max(bounds[param_idx][0], min(bounds[param_idx][1], neighbor[param_idx]))
                    neighbor_params.append(neighbor)

                # 并行计算多个邻域解的能量
                energies = pool.map(self.objective, neighbor_params)

                # 找到最佳邻域解
                min_energy_idx = np.argmin(energies)
                neighbor_energy = energies[min_energy_idx]
                neighbor = neighbor_params[min_energy_idx]

                # 计算能量差
                delta_energy = neighbor_energy - current_energy

                # 决定是否接受新解
                if delta_energy < 0 or random.random() < np.exp(-delta_energy / temperature):
                    current_solution = np.copy(neighbor)
                    current_energy = neighbor_energy

                    # 更新最佳解
                    if current_energy < best_energy:
                        best_solution = np.copy(current_solution)
                        best_energy = current_energy
                        stagnation_count = 0  # 重置停滞计数
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

                # 自适应调整步长
                if stagnation_count > 0 and stagnation_count % 50 == 0:
                    # 如果连续多次没有改进，减小步长
                    step_sizes = [step * 0.95 for step in step_sizes]
                    print(f"迭代 {iteration}: 自适应调整步长，当前平均步长: {np.mean(step_sizes):.6f}")

                # 降温
                temperature *= cooling_rate

                # 记录历史
                self.history.append((iteration, best_energy, temperature, np.mean(step_sizes)))

                # 早停检查
                best_energy_history.append(best_energy)
                if len(best_energy_history) > stagnation_threshold:
                    recent_best = best_energy_history[-stagnation_threshold:]
                    if len(set([round(e, 6) for e in recent_best])) <= 3:  # 如果最近的最优值变化很小
                        print(f"迭代 {iteration}: 算法已收敛，提前终止")
                        break

                # 打印进度
                if iteration % 100 == 0:
                    print(f"迭代 {iteration}/{max_iter}, 当前最佳能量: {best_energy:.4f}, 当前温度: {temperature:.4f}")

        self.best_params = best_solution
        self.best_energy = best_energy
        return best_solution, best_energy, self.history

    def run_optimization(self, initial_params, bounds, max_iter=1000, initial_temp=100, cooling_rate=0.95):
        # 运行优化并显示结果
        print("开始模拟退火优化...")
        best_params, best_energy, history = self.adaptive_simulated_annealing(
            initial_params, bounds,
            max_iter=max_iter,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate
        )

        # 计算预测结果和误差
        flow_pred = self.main_flow(self.t, best_params)
        rmse = np.sqrt(np.mean((flow_pred - self.flow) ** 2))
        mae = self.calculate_mae(flow_pred)
        print(f"优化完成！RMSE: {rmse:.6f}, MAE: {mae:.6f}")

        # 保存结果
        self._save_results(best_params, rmse, mae)

        # 绘制结果
        self._plot_results(best_params, history)

        # 进行灵敏度分析
        self.sensitivity_analysis(best_params)

        return best_params, rmse, mae

    def _save_results(self, best_params, rmse, mae):
        # 保存优化结果到文件
        # 创建结果文件夹
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(results_dir, f"result_{timestamp}.txt")

        # 计算指定时刻的车流量
        t_730 = 15  # 7:30对应t=15
        t_830 = 45  # 8:30对应t=45
        flow1_730, flow2_730, flow3_730, flow4_730 = self.branch_flow(np.array([t_730]), best_params)
        flow1_830, flow2_830, flow3_830, flow4_830 = self.branch_flow(np.array([t_830]), best_params)

        # 写入结果
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("优化结果：\n")
            f.write(f"支路1稳定流量 (a): {best_params[0]:.4f}\n")
            f.write(f"支路2参数 (b1, b2, b3): ({best_params[1]:.4f}, {best_params[2]:.4f}, {best_params[3]:.4f})\n")
            f.write(f"支路2转折点 (t_break1, t_break2): ({best_params[4]:.4f}, {best_params[5]:.4f})\n")
            f.write(f"支路3参数 (c1, c2, c3): ({best_params[6]:.4f}, {best_params[7]:.4f}, {best_params[8]:.4f})\n")
            f.write(f"支路3稳定开始时刻 (t_break3): {best_params[9]:.4f}\n")
            f.write(
                f"支路4参数 (d1, d2, d3, d4): ({best_params[10]:.4f}, {best_params[11]:.4f}, {best_params[12]:.4f}, {best_params[13]:.4f})\n")
            f.write(f"\nRMSE: {rmse:.6f}\n")
            f.write(f"\nMAE: {mae:.6f}\n")
            f.write("\n7:30时刻各支路车流量：\n")
            f.write(f"支路1: {flow1_730[0]:.2f}\n")
            f.write(f"支路2: {flow2_730[0]:.2f}\n")
            f.write(f"支路3: {flow3_730[0]:.2f}\n")
            f.write(f"支路4: {flow4_730[0]:.2f}\n")
            f.write("\n8:30时刻各支路车流量：\n")
            f.write(f"支路1: {flow1_830[0]:.2f}\n")
            f.write(f"支路2: {flow2_830[0]:.2f}\n")
            f.write(f"支路3: {flow3_830[0]:.2f}\n")
            f.write(f"支路4: {flow4_830[0]:.2f}\n")

            # 输出函数表达式
            a, b1, b2, b3, t_break1, t_break2, c1, c2, c3, t_break3, d1, d2, d3, d4 = best_params
            f.write("\n函数表达式：\n")
            f.write(f"支路1: f1(t) = {a:.4f}\n")
            f.write(f"支路2:\n")
            f.write(f"  当 t <= {t_break1:.1f} 时: f2(t) = {b1:.4f}*t + {b2:.4f}\n")
            f.write(f"  当 {t_break1:.1f} < t <= {t_break2:.1f} 时: f2(t) = {b3:.4f}\n")
            f.write(f"  当 t > {t_break2:.1f} 时: f2(t) = {b1:.4f}*(t-{t_break2:.1f}) + {b3:.4f}\n")
            f.write(f"支路3:\n")
            f.write(f"  当 t <= {t_break3:.1f} 时: f3(t) = {c1:.4f}*t + {c2:.4f}\n")
            f.write(f"  当 t > {t_break3:.1f} 时: f3(t) = {c3:.4f}\n")
            f.write(f"支路4: f4(t) = {d1:.4f}*sign(sin({d2:.4f}*t + {d3:.4f})) + {d4:.4f}\n")

        print(f"结果已保存到: {result_file}")

    def _plot_results(self, best_params, history):
        """绘制优化结果图表"""
        # 创建结果文件夹
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # 计算各支路流量
        flow1, flow2, flow3, flow4 = self.branch_flow(self.t, best_params)
        flow_pred = self.main_flow(self.t, best_params)

        # 绘制主路预测结果
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.flow, 'bo-', label='主路5实际车流量')
        plt.plot(self.t, flow_pred, 'r--', label='主路5预测车流量')
        plt.xlabel('时间t (相对于7:00的分钟数/2)')
        plt.ylabel('车流量')
        plt.title('主路5车流量预测结果')
        plt.grid(True)
        plt.legend()

        # 绘制各支路流量
        plt.subplot(2, 1, 2)
        plt.plot(self.t, flow1, 'g-', label='支路1车流量')
        plt.plot(self.t, flow2, 'm-', label='支路2车流量')
        plt.plot(self.t, flow3, 'c-', label='支路3车流量')
        plt.plot(self.t, flow4, 'y-', label='支路4车流量')
        plt.axvline(x=best_params[4], color='k', linestyle='--', label=f'支路2第一个转折点 (t={best_params[4]:.1f})')
        plt.axvline(x=best_params[5], color='k', linestyle='--', label=f'支路2第二个转折点 (t={best_params[5]:.1f})')
        plt.axvline(x=best_params[9], color='b', linestyle='--', label=f'支路3稳定开始时刻 (t={best_params[9]:.1f})')
        plt.xlabel('时间t (相对于7:00的分钟数/2)')
        plt.ylabel('车流量')
        plt.title('各支路车流量')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '主路5和各支路车流量.png'), dpi=300, bbox_inches='tight')

        # 绘制支路车流量叠加图
        plt.figure(figsize=(14, 8))
        plt.plot(self.t, self.flow, 'bo-', label='主路5实际车流量', alpha=0.5)
        plt.plot(self.t, flow_pred, 'r--', label='主路5预测车流量')
        flow1_delayed = np.zeros_like(self.t, dtype=float)
        flow2_delayed = np.zeros_like(self.t, dtype=float)
        flow1_delayed[self.t >= self.delay] = flow1[np.where(self.t >= self.delay)[0] - self.delay]
        flow2_delayed[self.t >= self.delay] = flow2[np.where(self.t >= self.delay)[0] - self.delay]
        flow1_delayed[self.t < self.delay] = flow1[0]
        flow2_delayed[self.t < self.delay] = flow2[0]
        plt.fill_between(self.t, 0, flow1_delayed, alpha=0.3, label='支路1车流量(延迟后)')
        plt.fill_between(self.t, flow1_delayed, flow1_delayed + flow2_delayed, alpha=0.3, label='支路2车流量(延迟后)')
        plt.fill_between(self.t, flow1_delayed + flow2_delayed, flow1_delayed + flow2_delayed + flow3, alpha=0.3,
                         label='支路3车流量')
        plt.fill_between(self.t, flow1_delayed + flow2_delayed + flow3, flow1_delayed + flow2_delayed + flow3 + flow4,
                         alpha=0.3, label='支路4车流量')
        plt.xlabel('时间t (相对于7:00的分钟数/2)')
        plt.ylabel('车流量')
        plt.title('支路车流量叠加及主路车流量比较')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '支路车流量叠加.png'), dpi=300, bbox_inches='tight')

        # 绘制优化过程
        plt.figure(figsize=(16, 6))
        iterations, energies, temps, steps = zip(*history)

        plt.subplot(1, 3, 1)
        plt.plot(iterations, energies, 'b-')
        plt.xlabel('迭代次数')
        plt.ylabel('误差 (MSE)')
        plt.title('模拟退火算法收敛过程')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(iterations, temps, 'r-')
        plt.xlabel('迭代次数')
        plt.ylabel('温度')
        plt.title('模拟退火算法温度下降过程')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(iterations, steps, 'g-')
        plt.xlabel('迭代次数')
        plt.ylabel('平均步长')
        plt.title('自适应步长调整过程')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '退火过程分析.png'), dpi=300, bbox_inches='tight')

        plt.show()

    def sensitivity_analysis(self, best_params):
        """灵敏度分析"""
        base_energy = self.objective(best_params)
        sensitivities = []
        param_names = ['q1', 'p1', 'r1', 'q2', 't_break1', 't_break2', 'p2', 'r2', 'q3', 't1', 'A1', 'ω1', 'φ1', 'k1']

        for i in range(len(best_params)):
            new_params = np.copy(best_params)
            # 增加参数值 5%
            new_params[i] *= 1.05
            new_energy = self.objective(new_params)
            sensitivity = (new_energy - base_energy) / base_energy
            sensitivities.append(sensitivity)

        # 绘制灵敏度分析柱状图
        plt.figure(figsize=(12, 6))
        plt.bar(param_names, sensitivities, color=(178/255, 221/255, 202/255))
        plt.xlabel('参数')
        plt.ylabel('灵敏度')
        plt.title('问题2：参数灵敏度分析')
        plt.grid(True)
        plt.tight_layout()

        # 创建结果文件夹
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        plt.savefig(os.path.join(results_dir, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # 数据文件路径
    excel_path = r"E:\AAD竞赛\数模\五一建模\2025-51MCM-Problem A\附件(Attachment).xlsx"

    # 初始化优化器
    optimizer = TrafficFlowOptimizer(data_path=excel_path)

    # 设定初始参数，固定 t_break1 和 t_break2
    initial_params = [
        20.0, 0.5, 5.0, 20.0, 24.0, 37.0, 0.6, 5.0, 25.0, 20.0, 5.0, 0.5, 0.0, 10.0
    ]

    # 设定参数边界，固定 t_break1 和 t_break2
    bounds = [
        (5.0, 30.0), (0.0, 2.0), (0.0, 20.0), (10.0, 30.0), (24.0, 24.0), (37.0, 37.0),
        (0.0, 2.0), (0.0, 20.0), (10.0, 40.0), (15.0, 25.0), (1.0, 15.0), (0.1, 1.0),
        (-np.pi, np.pi), (5.0, 20.0)
    ]

    # 运行优化
    best_params, rmse, mae = optimizer.run_optimization(
        initial_params, bounds,
        max_iter=1000,
        initial_temp=100,
        cooling_rate=0.95
    )
