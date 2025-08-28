import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.optimize import fsolve, root_scalar 
import math

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


class DragonRectangleSimulation:
    def __init__(self):
        # --- 1. 定义系统常量和参数 ---
        self.num_segments = 223
        self.first_segment_length = 286.0  # cm (龙头到第一节的距离)
        self.segment_length = 165.0  # cm (节点间距离)
        self.speed = 100.0  # cm/s

        # 阿基米德螺旋线参数 r = a * theta
        self.spiral_coeff = 55.0 / (2.0 * np.pi)

        # 仿真参数
        self.dt = 1.0  # 初始时间步长 (s)，会动态调整
        self.time = 0.0

        # --- 碰撞检测状态 ---
        self.collision_detected = False
        self.collision_time = -1.0
        self.collision_head_pos = None

        # --- 2. 矩形板凳龙的几何参数 ---
        # 根据实际尺寸设置矩形参数
        self.dragon_head_length = 341.0    # cm (龙头板长)
        self.dragon_body_length = 220.0    # cm (龙身和龙尾板长)
        self.board_width = 30.0            # cm (所有板宽)
        self.hole_diameter = 5.5           # cm (孔径)
        self.hole_distance_from_head = 27.5 # cm (孔心距离板头的距离)
        
        # 矩形尺寸（用于碰撞检测和显示）
        # 第一节（龙头）的矩形尺寸
        self.head_rect_length = self.dragon_head_length  # 341 cm
        self.head_rect_width = self.board_width          # 30 cm
        
        # 其他节（龙身和龙尾）的矩形尺寸
        self.body_rect_length = self.dragon_body_length  # 220 cm
        self.body_rect_width = self.board_width          # 30 cm


        # --- 3. 创建杆长数组 ---
        # self.lengths[i] 是连接第 i 个和第 (i+1) 个节点的杆长
        self.lengths = np.full(self.num_segments, self.segment_length)
        self.lengths[0] = self.first_segment_length

        # --- 4. 初始化仿真状态 ---
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

    def arc_length_from_origin(self, theta):
        """
        精确计算从原点到指定角度theta的螺旋线弧长。
        公式: s(theta) = a/2 * [theta*sqrt(1+theta^2) + asinh(theta)]
        """
        a = self.spiral_coeff
        # 使用 asinh(theta) 代替 log(theta + sqrt(theta**2 + 1))，数值上更稳定
        return 0.5 * a * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))

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

    def get_rectangle_corners(self, center_x, center_y, angle, segment_index):
        """
        计算矩形的四个顶点坐标
        center_x, center_y: 矩形中心坐标
        angle: 矩形的旋转角度（相对于螺旋线的切线方向）
        segment_index: 节段索引（0为龙头，1及以后为龙身/龙尾）
        返回: 四个顶点的坐标数组 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # 根据节段索引确定矩形尺寸
        if segment_index == 0:  # 龙头
            rect_length = self.head_rect_length  # 341 cm
            rect_width = self.head_rect_width    # 30 cm
        else:  # 龙身和龙尾
            rect_length = self.body_rect_length  # 220 cm
            rect_width = self.body_rect_width    # 30 cm
        
        # 矩形的半长和半宽
        half_length = rect_length / 2.0
        half_width = rect_width / 2.0
        
        # 在局部坐标系中的四个顶点（矩形中心为原点）
        local_corners = np.array([
            [-half_length, -half_width],  # 左下
            [half_length, -half_width],   # 右下
            [half_length, half_width],    # 右上
            [-half_length, half_width]    # 左上
        ])
        
        # 旋转矩阵
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # 旋转并平移到全局坐标系
        global_corners = np.dot(local_corners, rotation_matrix.T)
        global_corners[:, 0] += center_x
        global_corners[:, 1] += center_y
        
        return global_corners

    def project_polygon(self, axis, polygon):
        """将多边形投影到轴上，并返回最小/最大投影值"""
        min_proj = np.dot(polygon[0], axis)
        max_proj = min_proj
        for i in range(1, len(polygon)):
            proj = np.dot(polygon[i], axis)
            if proj < min_proj:
                min_proj = proj
            elif proj > max_proj:
                max_proj = proj
        return min_proj, max_proj

    def check_collision_sat(self, poly1, poly2):
        """
        使用分离轴定理 (SAT) 检查两个凸多边形之间的碰撞。
        poly1, poly2: 顶点坐标的numpy数组, shape (N, 2)。
        """
        polygons = [poly1, poly2]
        for polygon in polygons:
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                
                edge = p2 - p1
                # 轴是边的法线
                axis = np.array([-edge[1], edge[0]])
                
                min1, max1 = self.project_polygon(axis, poly1)
                min2, max2 = self.project_polygon(axis, poly2)
                
                # 如果在这个轴上没有重叠，则它们没有碰撞
                if max1 < min2 or max2 < min1:
                    return False
        
        # 如果所有轴都存在投影重叠，则多边形碰撞
        return True

    def check_for_collisions(self):
        """
        根据指定逻辑检查碰撞。
        返回 (是否碰撞, 内部矩形索引, 外部矩形索引)
        """
        all_rects = self.get_all_rectangles()
        
        # 需要检查的两个内部矩形
        rect_inner_0 = all_rects[0]
        rect_inner_1 = all_rects[1]
        
        # --- 确定要检查的外部矩形范围 ---
        theta1 = self.thetas[0]
        theta2 = self.thetas[2]  # 第二节段后的节点

        # 找到距离龙头和节点2一个整圈外的节点索引
        k1 = np.searchsorted(self.thetas, theta1 + 2 * np.pi)
        k2 = np.searchsorted(self.thetas, theta2 + 2 * np.pi)
        
        # 根据您的逻辑定义外部矩形的搜索范围
        start_idx = max(2, k1 - 2)
        end_idx = min(self.num_segments - 1, k2 + 2)

        for i in range(start_idx, end_idx + 1):
            rect_outer = all_rects[i]
            
            # 检查龙头 (矩形 0) 与外部矩形的碰撞
            if self.check_collision_sat(rect_inner_0, rect_outer):
                return True, 0, i
            
            # 检查第二个矩形 (矩形 1) 与外部矩形的碰撞
            if self.check_collision_sat(rect_inner_1, rect_outer):
                return True, 1, i
                
        return False, -1, -1

    def get_all_rectangles(self):
        """
        获取所有矩形板凳龙的顶点坐标。
        一个矩形（板凳）由两个连续的节点定义。
        返回: 所有矩形的顶点坐标列表。
        """
        rectangles = []
        
        # 一个节段由两个连续的节点定义。总共有 self.num_segments 个节段。
        for i in range(self.num_segments):
            # 获取节段两端的节点角度
            theta1 = self.thetas[i]
            theta2 = self.thetas[i+1]
            
            # 获取节点的笛卡尔坐标
            x1, y1 = self.get_spiral_xy(theta1)
            x2, y2 = self.get_spiral_xy(theta2)
            
            # 计算矩形中心（两个节点的中点）
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # 计算矩形的旋转角度（两个节点连线的角度）
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            # 获取矩形的顶点坐标。节段索引为 i。
            # segment_index=0 是龙头，1-221 是龙身，222 是龙尾。
            # get_rectangle_corners 函数内部会根据索引选择正确的板凳长度。
            corners = self.get_rectangle_corners(center_x, center_y, angle, i)
            rectangles.append(corners)
        
        return rectangles

    def step(self):
        """执行一个时间步的仿真"""
        # --- 动态调整时间步长 ---
        if self.time < 400.0:
            self.dt = 1.0
        elif self.time < 411.0:
            self.dt = 0.1
        elif self.time < 412.0:
            self.dt = 0.01
        else:
            self.dt = 0.0001

        # 如果仿真时间超过430s或已检测到碰撞，则停止
        if self.time >= 430.0 or self.collision_detected:
            return

        # --- 核心更新逻辑 ---
        
        # 1. 更新龙头的位置 (高精度方法)
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
            def find_theta_eq(theta_new):
                return self.arc_length_from_origin(theta_new) - s_new_target
            
            # 使用 root_scalar 求解器进行高精度求解
            sol = root_scalar(find_theta_eq, bracket=[0, theta_old], method='brentq', xtol=1e-15)
            
            if sol.converged:
                self.thetas[0] = sol.root
            else:
                print("警告: 高精度龙头位置求解失败！")
                return

        # 2. 依次更新后续所有节点的位置
        for i in range(self.num_segments):
            theta_prev = self.thetas[i]
            L_current = self.lengths[i]
            
            # 使用上一帧的角度作为本次求解的初始猜测值
            initial_guess = self.thetas[i+1]
            
            theta_next, = fsolve(self.distance_equation, initial_guess, 
                               args=(theta_prev, L_current, self.spiral_coeff))
            self.thetas[i+1] = theta_next
            
        self.time += self.dt

        # --- 在每个时间步后执行碰撞检测 ---
        collided, inner_idx, outer_idx = self.check_for_collisions()
        if collided:
            self.collision_detected = True
            self.collision_time = self.time
            self.collision_head_pos = self.get_spiral_xy(self.thetas[0])
            print("="*60)
            print(f"!!! 碰撞发生: 时间 t = {self.collision_time:.10f} s !!!")
            print(f"  -> 内部矩形 {inner_idx} 与 外部矩形 {outer_idx} 发生碰撞。")
            print(f"  -> 碰撞时龙头坐标: ({self.collision_head_pos[0]:.10f}, {self.collision_head_pos[1]:.10f})")
            print("="*60)

    def get_all_node_coords(self):
        """将所有节点的角度转换为笛卡尔坐标"""
        return self.get_spiral_xy(self.thetas)

    def plot_spiral_background(self, ax, theta_range_max):
        """绘制螺旋线背景"""
        theta_spiral_bg = np.linspace(0, theta_range_max, 20000)
        x_spiral_bg, y_spiral_bg = self.get_spiral_xy(theta_spiral_bg)
        ax.plot(x_spiral_bg, y_spiral_bg, '--', color='gray', linewidth=0.8, label='螺旋轨迹')


# --- Main execution ---
if __name__ == '__main__':
    # 创建仿真实例
    sim = DragonRectangleSimulation()

    # --- 快进到300秒 ---
    target_time = 300.0
    print(f"正在快进到 {target_time:.2f} 秒，请稍候...")
    next_print_time = 10.0
    while sim.time < target_time:
        sim.step()
        if sim.time >= next_print_time:
            print(f"已模拟到: {sim.time:.2f} s")
            next_print_time += 10.0
    print("快进完成！")


    # 设置Matplotlib进行动画展示
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 创建用于表示节点连接的Line2D对象
    line, = ax.plot([], [], 'o-', lw=1, markersize=2, color='red', alpha=0.7, label='节点连接')
    
    # 创建用于显示矩形的集合
    rectangles_collection = PatchCollection([], facecolors='lightblue', 
                                          edgecolors='blue', alpha=0.8, linewidths=1)
    ax.add_collection(rectangles_collection)
    
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
        rectangles_collection.set_paths([])
        return line, rectangles_collection

    def animate(frame):
        """动画每一帧的更新函数"""
        # 如果检测到碰撞，则停止动画
        if sim.collision_detected:
            ani.event_source.stop()
            return line, rectangles_collection, title

        sim.step()
        
        # --- 更新节点连接线 ---
        x_coords, y_coords = sim.get_all_node_coords()
        line.set_data(x_coords, y_coords)
        
        # --- 更新矩形显示 ---
        rectangles_data = sim.get_all_rectangles()
        
        # 创建矩形补丁对象
        patches = []
        for rect_corners in rectangles_data:
            # 使用Polygon创建矩形
            from matplotlib.patches import Polygon
            rect_patch = Polygon(rect_corners, closed=True)
            patches.append(rect_patch)
        
        # 更新矩形集合
        rectangles_collection.set_paths(patches)
        
        # 更新标题
        title.set_text(f"时间: {sim.time:.6f} s | 龙头圈数: {sim.thetas[0]/(2*np.pi):.6f}")
        
        return line, rectangles_collection, title

    # --- 计算总帧数 ---
    # 时间 < 350s: dt=1.0 -> 350 帧
    # 时间 >= 350s: dt=0.01 -> (430-350)/0.01 = 8000 帧
    total_frames = 350 + 50000

    # 创建并启动动画
    ani = FuncAnimation(fig, animate, frames=total_frames, init_func=init, 
                       blit=False, interval=10, repeat=True)
    
    # --- 添加暂停/继续功能 ---
    # 使用一个列表（可变对象）来存储暂停状态，以便在嵌套函数中修改
    pause_state = [False]
    def toggle_pause(event):
        # 检查按下的键是否是空格键
        if event.key == ' ':
            if pause_state[0]:
                # 如果已暂停，则继续动画
                ani.resume()
                print("动画已继续")
            else:
                # 如果正在运行，则暂停动画
                ani.pause()
                print("动画已暂停，按空格键继续...")
            # 切换暂停状态
            pause_state[0] = not pause_state[0]

    # 将键盘按下事件与我们的处理函数连接起来
    fig.canvas.mpl_connect('key_press_event', toggle_pause)
    
    plt.tight_layout()
    plt.show()