import numpy as np
from scipy.optimize import fsolve, root_scalar, minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle, Arc
from matplotlib.collections import PatchCollection

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DragonRectangleSimulation:
    """
    基于弧长的S形掉头轨迹仿真类
    """
    def __init__(self, pitch, debug=False):
        # --- 1. 定义系统常量和参数 ---
        self.debug = debug
        self.pitch = pitch
        self.spiral_coeff = self.pitch / (2.0 * np.pi)  # a = p / (2*pi)

        self.num_segments = 223
        self.first_segment_length = 286.0
        self.segment_length = 165.0
        self.speed = 100.0

        self.dt = 1
        self.time = -100.0
        
        # 暂停功能相关参数
        self.is_paused = False
        self.is_manual_paused = False  # 手动暂停状态
        self.last_pause_time = -5.0  # 第一次暂停的时间点
        self.pause_interval = 4.0  # 每4秒暂停一次
        self.next_pause_time = -5.0  # 下一次暂停的时间点
        self.pause_type = 'none'  # 暂停类型：'none', 'auto', 'manual'

        # --- 2. 矩形板凳龙的几何参数 ---
        self.dragon_head_length = 341.0
        self.dragon_body_length = 220.0
        self.board_width = 30.0
        
        self.head_rect_length = self.dragon_head_length
        self.head_rect_width = self.board_width
        self.body_rect_length = self.dragon_body_length
        self.body_rect_width = self.board_width

        # --- 3. S形掉头轨迹参数（单位：cm） ---
        self.R1 = 300.49  # 大圆半径
        self.R2 = 150.24  # 小圆半径
        self.O1 = np.array([-76.04, -130.61])  # 大圆圆心
        self.O2 = np.array([174.70, 243.94])   # 小圆圆心
        
        # 圆弧角度范围
        self.O1_start_angle = -2.2776 + 2 * np.pi
        self.O1_end_angle = 0.9809 # 约4.0056
        self.O2_start_angle = -2.1607
        self.O2_end_angle = 0.8735
        
        # --- 4. 创建杆长数组 ---
        self.lengths = np.full(self.num_segments, self.segment_length)
        self.lengths[0] = self.first_segment_length

        # --- 5. 计算关键点和轨迹 ---
        self.initialize_trajectory()
        
        # --- 6. 初始化位置 ---
        self.positions = []
        self.update_all_positions()
        
        # # 验证初始化（调试用）
        # if self.debug:
        #     self.verify_initialization()

    def initialize_trajectory(self):
        """初始化轨迹：计算D点、E点，离散化圆弧轨迹"""
        
        # 计算D点：盘入螺线与掉头空间的交点（r=450cm）
        self.radius_turn = 450.0  # 掉头空间半径
        self.theta_D = self.radius_turn / self.spiral_coeff
        self.D = np.array([self.radius_turn * np.cos(self.theta_D), 
                          self.radius_turn * np.sin(self.theta_D)])
        
        # E点与D点关于原点对称（盘出螺线是中心对称的）
        self.E = -self.D
        self.theta_E = self.theta_D  # 盘出螺线上对应的theta相同
        
        # 计算圆弧长度
        self.O1_arc_length = abs(self.O1_end_angle - self.O1_start_angle) * self.R1
        self.O2_arc_length = abs(self.O2_end_angle - self.O2_start_angle) * self.R2
        self.total_arc_length = self.O1_arc_length + self.O2_arc_length
        
        if self.debug:
            print(f"D点: ({self.D[0]:.5f}, {self.D[1]:.5f}), theta_D={self.theta_D:.5f}")
            print(f"E点: ({self.E[0]:.5f}, {self.E[1]:.5f})")
            print(f"O1圆弧长度: {self.O1_arc_length:.5f} cm")
            print(f"O2圆弧长度: {self.O2_arc_length:.5f} cm")
            print(f"总圆弧长度: {self.total_arc_length:.5f} cm")
        
        # 离散化圆弧轨迹（高精度）
        self.discretize_arc_trajectory()
        
        # 计算初始龙头位置对应的theta
        self.initial_theta = self.calculate_initial_theta()

    def calculate_initial_theta(self):
        """计算t=-100时龙头的theta"""
        # 从D点向外100秒的行程
        travel_distance = abs(self.time) * self.speed  # 10000 cm
        
        # D点的弧长
        arc_length_D = self.arc_length_from_origin(self.theta_D)
        
        # 初始位置的弧长
        initial_arc_length = arc_length_D + travel_distance
        
        # 通过弧长反算theta
        def arc_eq(theta):
            return self.arc_length_from_origin(theta) - initial_arc_length
        
        try:
            sol = root_scalar(arc_eq, bracket=[self.theta_D, 100], method='brentq',xtol=1e-10)
            return sol.root
        except:
            print("计算初始theta失败, 使用默认值")
            return 31.8618  # 使用默认值

    def discretize_arc_trajectory(self, num_points=10000):
        """离散化圆弧轨迹"""
        self.arc_points = []
        self.arc_distances = []  # 从D点沿圆弧的累积距离
        
        # O1圆弧上的点
        angles_O1 = np.linspace(self.O1_start_angle, self.O1_end_angle, int(num_points * 0.66))
        for angle in angles_O1:
            x = self.O1[0] + self.R1 * np.cos(angle)
            y = self.O1[1] + self.R1 * np.sin(angle)
            self.arc_points.append(np.array([x, y]))
            
            # 计算从起点的弧长
            arc_from_start = abs(angle - self.O1_start_angle) * self.R1
            self.arc_distances.append(arc_from_start)
        
        # O2圆弧上的点
        angles_O2 = np.linspace(self.O2_start_angle, self.O2_end_angle, int(num_points * 0.34))
        for angle in angles_O2:
            x = self.O2[0] + self.R2 * np.cos(angle)
            y = self.O2[1] + self.R2 * np.sin(angle)
            self.arc_points.append(np.array([x, y]))
            
            # 计算从起点的弧长（加上O1的总长度）
            arc_from_start = self.O1_arc_length + abs(angle - self.O2_start_angle) * self.R2
            self.arc_distances.append(arc_from_start)
        
        self.arc_points = np.array(self.arc_points)
        self.arc_distances = np.array(self.arc_distances)

    def get_spiral_xy(self, theta):
        """获取盘入螺线上的坐标"""
        r = self.spiral_coeff * theta
        return np.array([r * np.cos(theta), r * np.sin(theta)])
    
    def get_spiral_out_xy(self, theta):
        """获取盘出螺线上的坐标（中心对称）"""
        r = self.spiral_coeff * theta
        return np.array([-r * np.cos(theta), -r * np.sin(theta)])

    def arc_length_from_origin(self, theta):
        """计算从原点到theta的螺线弧长"""
        a = self.spiral_coeff
        return 0.5 * a * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))

    def get_head_position(self):
        """获取龙头的当前位置"""
        if self.time <= 0:
            # 龙头在盘入螺线上
            # 计算当前时刻龙头应该到达的弧长位置
            time_elapsed = abs(self.time)  # 从t=-100到当前时刻经过的时间
            travel_distance = time_elapsed * self.speed  # 已经行进的距离
            
            # D点的弧长
            arc_length_D = self.arc_length_from_origin(self.theta_D)
            
            # 当前龙头的弧长 = D点弧长 + 剩余要走的距离
            remaining_distance = abs(self.time) * self.speed  # 还要走多远才能到D点
            current_arc_length = arc_length_D + remaining_distance
            
            # 通过弧长找theta
            def arc_eq(theta):
                return self.arc_length_from_origin(theta) - current_arc_length
            
            try:
                # 当前theta应该大于theta_D（在外圈）
                sol = root_scalar(arc_eq, bracket=[self.theta_D, self.initial_theta+10], method='brentq')
                current_theta = sol.root
            except Exception as e:
                if self.debug:
                    print(f"计算当前theta失败: {e}, time={self.time}, target_arc_length={current_arc_length}")
                # 线性近似作为备用
                current_theta = self.theta_D + remaining_distance / (self.spiral_coeff * self.theta_D)
            
            return self.get_spiral_xy(current_theta), current_theta, 'spiral_in'
            
        else:
            # t > 0: 龙头在圆弧或盘出螺线
            travel_in_turn = self.speed * self.time
            
            if travel_in_turn <= self.total_arc_length:
                # 龙头在圆弧上
                # 找到最接近的离散点
                idx = np.searchsorted(self.arc_distances, travel_in_turn)
                if idx >= len(self.arc_points):
                    idx = len(self.arc_points) - 1
                
                # 插值以获得更精确的位置
                if idx > 0 and idx < len(self.arc_points):
                    t = (travel_in_turn - self.arc_distances[idx-1]) / (self.arc_distances[idx] - self.arc_distances[idx-1])
                    pos = (1-t) * self.arc_points[idx-1] + t * self.arc_points[idx]
                else:
                    pos = self.arc_points[idx]
                
                return pos, travel_in_turn, 'arc'
                
            else:
                # 龙头在盘出螺线上
                excess = travel_in_turn - self.total_arc_length
                
                # E点的弧长
                arc_length_E = self.arc_length_from_origin(self.theta_E)
                
                # 当前弧长
                current_arc_length = arc_length_E + excess
                
                # 通过弧长找theta
                def arc_eq(theta):
                    return self.arc_length_from_origin(theta) - current_arc_length
                
                try:
                    sol = root_scalar(arc_eq, bracket=[self.theta_E, 100], method='brentq')
                    current_theta = sol.root
                except:
                    current_theta = self.theta_E + excess / (self.spiral_coeff * self.theta_E)
                
                return self.get_spiral_out_xy(current_theta), current_theta, 'spiral_out'

    def bisection_solve(self, func, a, b, tol=1e-8, max_iter=100):
        """
        二分法求解方程 func(x) = 0
        :param func: 目标函数
        :param a: 区间左端点
        :param b: 区间右端点
        :param tol: 容差
        :param max_iter: 最大迭代次数
        :return: 解，如果找不到返回None
        """
        # 检查区间端点函数值符号是否相反
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            # print('fa fb >0')
            return None  # 区间内可能没有根
        
        for _ in range(max_iter):
            c = (a + b) / 2.0
            fc = func(c)
            
            if abs(fc) < tol or abs(b - a) < tol:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return (a + b) / 2.0  # 返回中点作为近似解

    @staticmethod
    def distance_equation_spiral(theta2, prev_pos, L, a):
        """
        螺线上两点距离约束方程。
        目标是找到一个 theta2，使得其与 prev_pos 对应的点之间的距离为 L。
        """
        r2 = a * theta2
        
        # 计算theta2对应的坐标
        cos2, sin2 = np.cos(theta2), np.sin(theta2)
        x2 = r2 * cos2
        y2 = r2 * sin2
        
        # 计算距离平方差
        dx = prev_pos[0] - x2
        dy = prev_pos[1] - y2
        dist_sq = dx * dx + dy * dy
        L_sq = L * L
        
        return dist_sq - L_sq

    def find_next_node_on_spiral_in(self, prev_pos, L):
        """在盘入螺线上找下一个节点"""
        # 从前一个节点的位置反算theta1
        r_prev = np.linalg.norm(prev_pos)
        theta_prev = r_prev / self.spiral_coeff
        
        # 定义目标函数
        def target_func(theta):
            return self.distance_equation_spiral(theta, prev_pos, L, self.spiral_coeff)
        
        # 二分法求解：区间 [theta_prev, theta_prev + pi/2]
        a1 = max(theta_prev, self.theta_D)
        b1 = a1 + np.pi / 2
        
        theta_sol = self.bisection_solve(target_func, a1, b1)
        
        if theta_sol is not None and theta_sol > self.theta_D:
            return self.get_spiral_xy(theta_sol), theta_sol, 'spiral_in'
        
        if self.debug:
            print(f"二分法求解失败，theta_prev={theta_prev:.6f}, theta_D={self.theta_D:.6f}")
        
        return prev_pos, 0, 'unknown'

    def find_next_node_on_arc(self, prev_pos, L):
        """在圆弧上找下一个节点"""
        # 首先找到前一个节点在圆弧上的位置
        distances_to_prev = np.linalg.norm(self.arc_points - prev_pos, axis=1)
        prev_idx = np.argmin(distances_to_prev)
        prev_arc_distance = self.arc_distances[prev_idx]
        
        # 如果前一个节点距离为total_arc_length，说明是从盘出螺线来的
        if abs(prev_arc_distance - self.total_arc_length) < 5.0:
            # 从盘出螺线来的情况，可以搜索整个圆弧
            search_indices = np.arange(len(self.arc_points))
        else:
            # 正常情况，只搜索在前一个节点"之前"的点（arc_distance更小的点）
            search_indices = np.where(self.arc_distances < prev_arc_distance)[0]
        
        if len(search_indices) == 0:
            return None, 0, 'not_found'
        
        # 在搜索范围内找距离最接近L的点
        search_points = self.arc_points[search_indices]
        distances_to_prev_filtered = np.linalg.norm(search_points - prev_pos, axis=1)
        
        # 找到距离最接近L的点
        best_idx_in_filtered = np.argmin(np.abs(distances_to_prev_filtered - L))
        actual_idx = search_indices[best_idx_in_filtered]
        
        if abs(distances_to_prev_filtered[best_idx_in_filtered] - L) < 1.0:  # 容差1cm
            return self.arc_points[actual_idx], self.arc_distances[actual_idx], 'arc'
        
        return None, 0, 'not_found'

    @staticmethod  
    def distance_equation_spiral_out(theta2, prev_pos, L, a):
        """
        盘出螺线上两点距离约束方程。
        盘出螺线是盘入螺线的中心对称。
        """
        r2 = a * theta2
        
        # 盘出螺线是中心对称的，计算theta2对应的坐标
        cos2, sin2 = np.cos(theta2), np.sin(theta2)
        x2 = -r2 * cos2  # 盘出螺线坐标取负
        y2 = -r2 * sin2
        
        # 计算距离平方差
        dx = prev_pos[0] - x2
        dy = prev_pos[1] - y2
        dist_sq = dx * dx + dy * dy
        L_sq = L * L
        
        return dist_sq - L_sq

    def find_next_node_on_spiral_out(self, prev_pos, L):
        """在盘出螺线上找下一个节点"""
        # 从前一个节点的位置反算theta1（注意盘出螺线是中心对称的）
        r_prev = np.linalg.norm(prev_pos)
        theta_prev = r_prev / self.spiral_coeff
        
        # 定义目标函数
        def target_func(theta):
            return self.distance_equation_spiral_out(theta, prev_pos, L, self.spiral_coeff)
        
        # 二分法求解：区间 [max(theta_prev - pi/2, self.theta_D), theta_prev]
        # 盘出螺线的theta是递减的，所以区间是从较小值到较大值
        a1 = max(theta_prev - np.pi / 2, self.theta_D)
        b1 = theta_prev 
        
        if a1 >= b1:  # 确保区间有效
            if self.debug:
                print(f"盘出螺线二分法区间无效: a1={a1:.6f}, b1={b1:.6f}")
            return prev_pos, 0, 'unknown'
        
        theta_sol = self.bisection_solve(target_func, a1, b1)
        
        if theta_sol is not None and theta_sol > self.theta_D:
            return self.get_spiral_out_xy(theta_sol), theta_sol, 'spiral_out'
        
        # if self.debug:
        #     print(f"盘出螺线二分法求解失败，theta_prev={theta_prev:.6f}, theta_D={self.theta_D:.6f}")
        
        return prev_pos, 0, 'unknown'

    def find_next_node(self, prev_pos, prev_phase, L):
        """根据前一个节点找下一个节点"""
        # 根据前一个节点的phase，依次尝试不同的轨迹
        
        if prev_phase == 'spiral_in':
            # 先尝试在螺线上
            pos, param, phase = self.find_next_node_on_spiral_in(prev_pos, L)
            if phase != 'unknown':
                return pos, param, phase
            else: print("spiral_in 找不到下一个节点")
                    
        elif prev_phase == 'arc':
            # 先尝试在圆弧上
            pos, param, phase = self.find_next_node_on_arc(prev_pos, L)
            if phase == 'arc':
                return pos, param, phase
            
            # 否则可能还在螺线上（接近D点）
            if np.linalg.norm(prev_pos - self.D) < L + 50:
                pos, param, phase = self.find_next_node_on_spiral_in(prev_pos, L)
                if phase != 'unknown':
                    return pos, param, phase
                    
        elif prev_phase == 'spiral_out':
            # 在盘出螺线上
            pos, param, phase = self.find_next_node_on_spiral_out(prev_pos, L)
            if phase != 'unknown':
                return pos, param, phase
            
            # 可能还在圆弧上
            if np.linalg.norm(prev_pos - self.E) < L + 50:
                pos, param, phase = self.find_next_node_on_arc(prev_pos, L)
                if phase == 'arc':
                    return pos, param, phase
        
        # 如果都找不到，返回原位置
        return prev_pos, 0, 'unknown'

    def verify_initialization(self):
        """验证初始化是否正确"""
        print("\n=== 初始化验证 ===")
        print(f"总节点数: {len(self.positions)}")
        
        # 检查前几个节点的距离
        for i in range(min(5, len(self.positions)-1)):
            pos1 = self.positions[i]
            pos2 = self.positions[i+1]
            actual_dist = np.linalg.norm(pos2 - pos1)
            expected_dist = self.lengths[i]
            error = abs(actual_dist - expected_dist)
            
            print(f"杆 {i}: 期望距离={expected_dist:.2f}cm, 实际距离={actual_dist:.2f}cm, 误差={error:.2f}cm")
            
            if error > 1.0:  # 误差超过1cm
                print(f"  ⚠️ 警告: 杆 {i} 距离误差过大!")
        
        # 检查龙头位置
        head_pos = self.positions[0]
        print(f"龙头位置: ({head_pos[0]:.2f}, {head_pos[1]:.2f})")
        print(f"龙头相位: {self.node_phases[0]}")
        print("================\n")

    def update_all_positions(self):
        """更新所有节点的当前位置"""
        self.positions = []
        self.node_phases = []
        
        # 龙头位置
        head_pos, head_param, head_phase = self.get_head_position()
        self.positions.append(head_pos)
        self.node_phases.append(head_phase)
        
        # 后续节点
        for i in range(self.num_segments):
            prev_pos = self.positions[-1]
            prev_phase = self.node_phases[-1]
            L = self.lengths[i]
            
            next_pos, next_param, next_phase = self.find_next_node(prev_pos, prev_phase, L)
            self.positions.append(next_pos)
            self.node_phases.append(next_phase)

    def step(self):
        """仿真步进"""
        # if self.time > -5.0:
        #     self.dt = 0.5
        # if self.time > 0.1:
        #     self.dt = 0.1

        if self.time >= 100.0:
            return
        
        # 检查是否处于任何暂停状态
        if self.is_paused or self.is_manual_paused:
            return
        
        # # 检查是否需要自动暂停（当t > -5s时，每4秒暂停一次）
        # if self.time > -5.0 and self.time >= self.next_pause_time:
        #     self.is_paused = True
        #     self.pause_type = 'auto'
        #     self.next_pause_time += self.pause_interval
        #     print(f"自动暂停！当前时间: {self.time:.1f}s，按空格键继续...")
        #     return
        
        self.time += self.dt
        self.update_all_positions()
    
    def resume(self):
        """恢复运动（仅用于自动暂停）"""
        if self.is_paused and self.pause_type == 'auto':
            self.is_paused = False
            self.pause_type = 'none'
            print(f"继续运动！当前时间: {self.time:.1f}s")
    
    def toggle_manual_pause(self):
        """切换手动暂停状态"""
        self.is_manual_paused = not self.is_manual_paused
        if self.is_manual_paused:
            self.pause_type = 'manual'
            print(f"手动暂停！当前时间: {self.time:.1f}s")
        else:
            if self.pause_type == 'manual':
                self.pause_type = 'none'
            print(f"取消手动暂停！当前时间: {self.time:.1f}s")

    def get_rectangle_corners(self, center_x, center_y, angle, segment_index):
        """获取矩形的四个顶点"""
        rect_length = self.head_rect_length if segment_index == 0 else self.body_rect_length
        rect_width = self.board_width
        
        half_length = rect_length / 2.0
        half_width = rect_width / 2.0
        
        local_corners = np.array([
            [-half_length, -half_width], [half_length, -half_width],
            [half_length, half_width], [-half_length, half_width]
        ])
        
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        global_corners = np.dot(local_corners, rotation_matrix.T) + [center_x, center_y]
        return global_corners

    def get_all_rectangles(self):
        """获取所有矩形的顶点"""
        rectangles = []
        
        for i in range(min(self.num_segments, len(self.positions) - 1)):
            pos1 = self.positions[i]
            pos2 = self.positions[i+1]
            center_x, center_y = (pos1[0] + pos2[0]) / 2.0, (pos1[1] + pos2[1]) / 2.0
            angle = np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])
            corners = self.get_rectangle_corners(center_x, center_y, angle, i)
            rectangles.append(corners)
        
        return rectangles

    def plot_spiral_background(self, ax):
        """绘制背景轨迹"""
        # 绘制螺旋线
        theta_range = np.linspace(0, max(self.initial_theta, 40), 2000)
        r_range = self.spiral_coeff * theta_range
        x_spiral = r_range * np.cos(theta_range)
        y_spiral = r_range * np.sin(theta_range)
        ax.plot(x_spiral, y_spiral, '--', color='gray', linewidth=0.8, label='盘入螺旋轨迹')
        ax.plot(-x_spiral, -y_spiral, '--', color='gold', linewidth=0.8, label='盘出螺旋轨迹')
        
        # 绘制掉头圆弧
        arc_o1 = Arc((self.O1[0], self.O1[1]), 2 * self.R1, 2 * self.R1,
                     angle=0, theta1=np.degrees(self.O1_end_angle),
                     theta2=np.degrees(self.O1_start_angle), 
                     color='red', linewidth=2, label='掉头圆弧O1')
        ax.add_patch(arc_o1)
        
        arc_o2 = Arc((self.O2[0], self.O2[1]), 2 * self.R2, 2 * self.R2,
                     angle=0, theta1=np.degrees(self.O2_start_angle),
                     theta2=np.degrees(self.O2_end_angle),
                     color='blue', linewidth=2, label='掉头圆弧O2')
        ax.add_patch(arc_o2)
        
        # 标记关键点
        ax.plot(self.D[0], self.D[1], 'go', markersize=8, label=f'D点({self.D[0]:.1f}, {self.D[1]:.1f})')
        ax.plot(self.E[0], self.E[1], 'mo', markersize=8, label=f'E点({self.E[0]:.1f}, {self.E[1]:.1f})')
        ax.plot(self.O1[0], self.O1[1], 'ro', markersize=5)
        ax.plot(self.O2[0], self.O2[1], 'bo', markersize=5)


def run_animation_for_pitch(pitch, verbose=False, debug=False):
    """运行动画仿真"""
    if verbose:
        print(f"--- 正在可视化S形掉头轨迹 p = {pitch:.4f} ---")

    sim = DragonRectangleSimulation(pitch=pitch, debug=debug)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title(f'S形掉头轨迹仿真（基于弧长）')
    
    # 键盘事件处理
    def on_key_press(event):
        if event.key == ' ':  # 空格键
            if sim.is_paused and sim.pause_type == 'auto':
                # 如果是自动暂停，恢复运动
                sim.resume()
            else:
                # 否则切换手动暂停状态
                sim.toggle_manual_pause()
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    line, = ax.plot([], [], 'o-', lw=1, markersize=2, color='green', alpha=0.7, label='节点连接')
    rectangles_collection = PatchCollection([], facecolors='lightblue', edgecolors='blue', alpha=0.6)
    ax.add_collection(rectangles_collection)
    title = ax.set_title("初始化中...")
    ax.set_aspect('equal')
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.grid(True, alpha=0.3)

    # 添加掉头空间圆
    circle = plt.Circle((0, 0), 450, color='purple', fill=False, linestyle='-.', linewidth=2, label='掉头空间(半径450cm)')
    ax.add_patch(circle)

    def init():
        sim.plot_spiral_background(ax)
        max_r = 1500
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return line, rectangles_collection, title

    def animate(frame):
        sim.step()
        
        # 更新节点连线
        if len(sim.positions) > 0:
            positions = np.array(sim.positions)
            line.set_data(positions[:, 0], positions[:, 1])
        
        # 更新矩形
        try:
            rectangles = sim.get_all_rectangles()
            patches = [Polygon(corners, closed=True) for corners in rectangles]
            rectangles_collection.set_paths(patches)
        except Exception as e:
            if debug:
                print(f"矩形更新失败: {e}")
        
        # 更新标题
        head_phase = sim.node_phases[0] if len(sim.node_phases) > 0 else 'unknown'
        phase_text = {
            'spiral_in': '盘入阶段',
            'arc': '掉头阶段',
            'spiral_out': '盘出阶段',
            'unknown': '未知'
        }.get(head_phase, '未知')
        
        # 显示暂停状态
        pause_text = ""
        if sim.is_paused and sim.pause_type == 'auto':
            pause_text = " [自动暂停 - 按空格继续]"
        elif sim.is_manual_paused:
            pause_text = " [手动暂停 - 按空格恢复]"
        
        title.set_text(f"螺距p={pitch:.1f}cm | {phase_text} | 时间:{sim.time:.1f}s{pause_text}")
        return line, rectangles_collection, title

    ani = FuncAnimation(fig, animate, init_func=init, blit=False, interval=1, repeat=True)
    
    plt.tight_layout()
    plt.show()


def turn_round():
    """主函数"""
    PITCH = 170.0
    
    print("="*80)
    print(f"S形掉头轨迹仿真（基于弧长）")
    print(f"螺距 p = {PITCH:.2f} cm")
    print(f"掉头空间半径: 450 cm")
    print("控制说明：")
    print("  - 自动暂停：当t > -5s时，每4秒自动暂停")
    print("  - 手动暂停：随时按空格键暂停/恢复画面")
    print("  - 自动暂停时按空格继续，其他时候按空格切换手动暂停")
    print("="*80)
    
    run_animation_for_pitch(PITCH, verbose=True, debug=True)


if __name__ == "__main__":
    turn_round()