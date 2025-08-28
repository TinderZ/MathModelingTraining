import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve, root_scalar 
import math

class DragonSimulation:
    def __init__(self):
        # --- 1. 定义系统常量和参数 ---
        self.num_segments = 223
        self.first_segment_length = 286.0  # cm
        self.segment_length = 165.0  # cm
        self.speed = 100.0  # cm/s

        # 阿基米德螺旋线参数 r = a * theta
        self.spiral_coeff = 55.0 / (2.0 * np.pi)

        # 仿真参数
        self.dt = 0.1  # 时间步长 (s)
        self.time = 0.0

        # --- 2. 创建杆长数组 ---
        # self.lengths[i] 是连接第 i 个和第 (i+1) 个节点的杆长
        self.lengths = np.full(self.num_segments, self.segment_length)
        self.lengths[0] = self.first_segment_length

        # --- 3. 初始化仿真状态 ---
        # 初始化存储所有节点角度的数组
        # self.thetas[i] 代表第 i 个节点的角度 (i=0 是龙头)
        self.thetas = np.zeros(self.num_segments + 1)
        
        # 初始时刻，龙头在第16圈
        self.thetas[0] = 16.0 * 2.0 * np.pi
        
        print("正在计算初始状态，这可能需要一些时间...")
        self.initialize_thetas()
        print("初始状态计算完成！")

    def get_spiral_xy(self, theta):
        """根据角度theta计算在阿基米德螺旋线上的笛卡尔坐标(x, y)"""
        r = self.spiral_coeff * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def arc_length_from_origin(self, theta):
        """
        【新增】精确计算从原点到指定角度theta的螺旋线弧长。
        公式: s(theta) = a/2 * [theta*sqrt(1+theta^2) + asinh(theta)]
        """
        a = self.spiral_coeff
        # 使用 asinh(theta) 代替 ln(theta + sqrt(theta**2 + 1))，数值上更稳定
        return 0.5 * a * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))
    
    @staticmethod
    def distance_equation(theta2, theta1, L, a):
        """
        运动学约束方程。
        目标是找到一个 theta2，使得其与 theta1 对应的点之间的距离为 L。
        f(theta2) = (两点间距离)^2 - L^2 = 0
        """
        # 使用高精度计算
        r1 = a * theta1
        r2 = a * theta2
        
        # 计算坐标差
        cos1, sin1 = np.cos(theta1), np.sin(theta1)
        cos2, sin2 = np.cos(theta2), np.sin(theta2)
        
        dx = r1 * cos1 - r2 * cos2
        dy = r1 * sin1 - r2 * sin2
        
        # 避免直接计算平方差，使用更稳定的形式
        dist_sq = dx * dx + dy * dy
        L_sq = L * L
        
        return dist_sq - L_sq



    def initialize_thetas(self):
        """根据第一个节点的位置，依次计算出所有其他节点的位置"""
        for i in range(self.num_segments):
            theta_prev = self.thetas[i]
            L_current = self.lengths[i]
            
            # 为 fsolve 提供一个合理的初始猜测值
            if theta_prev > 1:
                initial_guess_theta_next = theta_prev + L_current / (self.spiral_coeff * theta_prev)
            else:
                initial_guess_theta_next = theta_prev + 0.1

            theta_next, = fsolve(self.distance_equation,
                                 initial_guess_theta_next,
                                 args=(theta_prev, L_current, self.spiral_coeff))
            
            self.thetas[i+1] = theta_next
            
            if (i + 1) % 20 == 0:
                print(f"已初始化 {i+1}/{self.num_segments} 个后续节点...")

    def step(self):
        """执行一个时间步"""

        
        # 计算当前龙头位置距离原点的总弧长
        theta_old = self.thetas[0]
        s_old = self.arc_length_from_origin(theta_old)
        
        # 在 dt 时间内，龙头向内移动的距离
        delta_s = self.speed * self.dt
        
        # 计算移动后的目标弧长 (从原点算起)
        s_new_target = s_old - delta_s
        
        # 如果目标弧长小于等于0，说明已经到达或超过中心点
        if s_new_target <= 0:
            self.thetas[0] = 0
        else:
            # 我们需要找到一个新的角度 theta_new，使得 arc_length(theta_new) == s_new_target
            # 这等价于求解方程: arc_length(theta_new) - s_new_target = 0
            def find_theta_eq(theta_new):
                return self.arc_length_from_origin(theta_new) - s_new_target
            
            # 使用 root_scalar 求解器进行高精度求解。
            # 我们知道解在 [0, theta_old] 区间内，可以使用 brentq 方法，它快速且可靠。
            # xtol=1e-9 确保求解精度高于用户要求的8位小数。
            sol = root_scalar(find_theta_eq, bracket=[0, theta_old], method='brentq', xtol=1e-15)
            
            if sol.converged:
                self.thetas[0] = sol.root
            else:
                # 如果求解失败，打印警告。在正常情况下不应发生。
                print("警告: 高精度龙头位置求解失败！")
                return

        # 依次更新后续所有节点的位置
        for i in range(self.num_segments):
            theta_prev = self.thetas[i]
            L_current = self.lengths[i]
            
            # 使用上一帧的角度作为本次求解的初始猜测值
            initial_guess = self.thetas[i+1]
            
            theta_next, = fsolve(self.distance_equation, initial_guess, args=(theta_prev, L_current, self.spiral_coeff))
            self.thetas[i+1] = theta_next
            
        self.time += self.dt

    def get_all_node_coords(self):
        """将所有节点的角度转换为笛卡尔坐标"""
        return self.get_spiral_xy(self.thetas)

    def plot_spiral_background(self, ax, theta_range_max):
        """绘制螺旋线背景"""
        theta_spiral_bg = np.linspace(0, theta_range_max, 20000)
        x_spiral_bg, y_spiral_bg = self.get_spiral_xy(theta_spiral_bg)
        ax.plot(x_spiral_bg, y_spiral_bg, '--', color='gray', linewidth=0.8, label='螺线轨迹')


# --- Main execution ---
if __name__ == '__main__':
    # 创建仿真实例
    sim = DragonSimulation()

    # 设置Matplotlib进行动画展示
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 创建用于表示杆子链的Line2D对象
    line, = ax.plot([], [], 'o-', lw=2, markersize=3, color='royalblue')
    title = ax.set_title("时间: 0.00 s")
    
    # 设置坐标轴
    ax.set_aspect('equal')
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.grid(True)
    
    def init():
        """动画初始化函数"""
        # 绘制背景螺旋线
        sim.plot_spiral_background(ax, sim.thetas[-1] * 1.1)
        
        # 动态确定绘图范围
        max_r = sim.spiral_coeff * sim.thetas[-1] * 1.1
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.legend()
        line.set_data([], [])
        return line,

    def animate(frame):
        """动画每一帧的更新函数"""
        sim.step()
        
        # --- 更新绘图数据 ---
        x_coords, y_coords = sim.get_all_node_coords()
        line.set_data(x_coords, y_coords)
        
        # 更新标题
        title.set_text(f"时间: {sim.time:.2f} s | 龙头圈数: {sim.thetas[0]/(2*np.pi):.2f}")
        
        return line, title

    # 创建并启动动画
    ani = FuncAnimation(fig, animate, frames=3100, init_func=init, blit=True, interval=50, repeat=False)
    
    plt.tight_layout()
    plt.show()