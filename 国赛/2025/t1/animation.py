import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 1. 类定义 ---

class KinematicObject:
    """运动学物体的基类"""
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.trajectory = [self.pos.copy()]

    def update(self, dt):
        # 这个方法将在子类中被重写
        pass

    def log_position(self):
        self.trajectory.append(self.pos.copy())

class Missile(KinematicObject):
    """导弹类，匀速直线运动"""
    def __init__(self, initial_pos, target_pos, speed):
        super().__init__(initial_pos)
        self.target_pos = np.array(target_pos, dtype=float)
        self.speed = float(speed)
        
        direction = self.target_pos - self.pos
        self.velocity = direction / np.linalg.norm(direction) * self.speed

    def update(self, dt):
        self.pos += self.velocity * dt
        self.log_position()

class Drone(KinematicObject):
    """无人机类，等高度匀速直线飞行"""
    def __init__(self, initial_pos, target_pos, speed):
        super().__init__(initial_pos)
        target_pos_2d = np.array([target_pos[0], target_pos[1]])
        pos_2d = np.array([self.pos[0], self.pos[1]])
        
        direction_2d = target_pos_2d - pos_2d
        unit_direction_2d = direction_2d / np.linalg.norm(direction_2d)
        velocity_2d = unit_direction_2d * speed
        self.velocity = np.array([velocity_2d[0], velocity_2d[1], 0.0])

    def update(self, dt):
        self.pos += self.velocity * dt
        self.log_position()

class SmokeBomb(KinematicObject):
    """烟幕弹（下落过程），做斜抛运动"""
    def __init__(self, initial_pos, initial_velocity):
        super().__init__(initial_pos)
        self.initial_velocity = np.array(initial_velocity, dtype=float)
        self.initial_pos = np.array(initial_pos, dtype=float)
        self.g = 9.80665  # 重力加速度大小
        self.start_time = 0  # 记录开始时间，用于精确计算
        self.elapsed_time = 0  # 已运行时间

    def update(self, dt):
        self.elapsed_time += dt
        
        # 使用精确的物理公式计算位置
        # x(t) = x0 + v0x * t
        # y(t) = y0 + v0y * t  
        # z(t) = z0 + v0z * t - 0.5 * g * t^2
        
        self.pos[0] = self.initial_pos[0] + self.initial_velocity[0] * self.elapsed_time
        self.pos[1] = self.initial_pos[1] + self.initial_velocity[1] * self.elapsed_time
        self.pos[2] = self.initial_pos[2] + self.initial_velocity[2] * self.elapsed_time - 0.5 * self.g * self.elapsed_time**2
        
        self.log_position()

class SmokeCloud(KinematicObject):
    """烟幕云团类，匀速下沉"""
    def __init__(self, detonation_pos):
        super().__init__(detonation_pos)
        self.radius = 10.0
        self.sink_velocity = np.array([0.0, 0.0, -3.0])

    def update(self, dt):
        self.pos += self.sink_velocity * dt
        self.log_position()

# --- 2. 几何体绘制函数 ---

def draw_cylinder(ax, center, radius, height, color='cyan', alpha=0.3):
    """绘制圆柱体（真目标）"""
    # 圆柱体的底面和顶面
    theta = np.linspace(0, 2*np.pi, 30)
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    
    # 底面
    z_bottom = np.full_like(x_circle, center[2])
    ax.plot(x_circle, y_circle, z_bottom, color=color, linewidth=2)
    
    # 顶面
    z_top = np.full_like(x_circle, center[2] + height)
    ax.plot(x_circle, y_circle, z_top, color=color, linewidth=2)
    
    # 侧面线条
    for i in range(0, len(theta), 5):
        ax.plot([x_circle[i], x_circle[i]], 
                [y_circle[i], y_circle[i]], 
                [z_bottom[i], z_top[i]], 
                color=color, linewidth=1, alpha=alpha)

def draw_sphere(ax, center, radius, color='gray', alpha=0.3):
    """绘制球体（烟幕云团）"""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=alpha)

# --- 3. 参数初始化 ---

# 根据题目信息设置常量
# 目标位置 - 修正为真目标（圆柱体下底面圆心）
TRUE_TARGET_POS = np.array([0, 200, 0])  # 圆柱体下底面圆心
FALSE_TARGET_POS = np.array([0, 0, 0])   # 导弹实际瞄准的假目标

# 圆柱体参数
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0

# 导弹M1参数 - 朝向原点运动
M1_INITIAL_POS = (20000, 0, 2000)
M1_SPEED = 300

# 无人机FY1参数
FY1_INITIAL_POS = (17800, 0, 1800)
FY1_SPEED = 120

# 烟幕弹事件时间
T_DROP = 1.5
T_DETONATE = T_DROP + 3.6
T_EFFECT_END = T_DETONATE + 20

# 仿真参数
DT = 0.1  # 初始时间步长 (秒)，起爆后会变为0.01s
SIM_END_TIME = 12.1  # 总仿真时长 (秒)
time_steps = np.arange(0, SIM_END_TIME, DT)

# --- 4. 动画类 ---

class KinematicAnimation:
    def __init__(self):
        # 创建物体实例
        self.missile_m1 = Missile(M1_INITIAL_POS, FALSE_TARGET_POS, M1_SPEED)
        self.drone_fy1 = Drone(FY1_INITIAL_POS, FALSE_TARGET_POS, FY1_SPEED)
        
        # 用于存储烟幕弹和云团的变量
        self.Bomb = None
        self.cloud = None
        self.detonation_point = None
        
        # 遮挡逻辑相关变量
        self.is_occluded = False  # 当前是否被遮挡
        self.occlusion_start_time = None  # 遮挡开始时间
        self.occlusion_end_time = None  # 遮挡结束时间
        self.total_occlusion_time = 0  # 总遮挡时间
        self.occlusion_periods = []  # 存储所有遮挡时间段 [(start, end), ...]
        self.occlusion_ratio = 0.0  # 当前遮挡比例
        
        # 时间变量和仿真控制
        self.current_time = 0
        self.frame_count = 0
        self.simulation_running = True  # 仿真运行状态
        self.is_paused = False  # 动画暂停状态
        self.dt = DT  # 动态时间步长
        
        # 设置图形
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 设置交互模式
        plt.ion()  # 打开交互模式
        
        # 初始化绘图元素
        self.init_plot_elements()
        
    def init_plot_elements(self):
        """初始化绘图元素"""
        # 导弹轨迹
        self.missile_line, = self.ax.plot([], [], [], 'r-', linewidth=2, label='导弹M1轨迹')
        self.missile_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='导弹M1')
        
        # 无人机轨迹
        self.drone_line, = self.ax.plot([], [], [], 'b--', linewidth=2, label='无人机FY1轨迹')
        self.drone_point, = self.ax.plot([], [], [], 'bs', markersize=6, label='无人机FY1')
        
        # 烟幕弹轨迹
        self.Bomb_line, = self.ax.plot([], [], [], 'g:', linewidth=2, label='烟幕弹轨迹')
        self.Bomb_point, = self.ax.plot([], [], [], 'go', markersize=5, label='烟幕弹')
        
        # 烟幕云团轨迹
        self.cloud_line, = self.ax.plot([], [], [], 'm-', linewidth=2, alpha=0.7, label='烟幕云团轨迹')
        self.cloud_point, = self.ax.plot([], [], [], 'mo', markersize=12, alpha=0.7, label='烟幕云团')
        
        # 标记关键点
        self.ax.scatter(*M1_INITIAL_POS, color='red', s=100, marker='^', label='M1起点')
        self.ax.scatter(*FY1_INITIAL_POS, color='blue', s=100, marker='^', label='FY1起点')
        self.ax.scatter(*FALSE_TARGET_POS, color='black', s=150, marker='x', label='假目标(原点)')
        
        # 绘制圆柱体目标
        draw_cylinder(self.ax, TRUE_TARGET_POS, CYLINDER_RADIUS, CYLINDER_HEIGHT, 'cyan', 0.3)
        self.ax.text(TRUE_TARGET_POS[0], TRUE_TARGET_POS[1], TRUE_TARGET_POS[2] + CYLINDER_HEIGHT + 5, 
                    '真目标\n(圆柱体)', fontsize=10, ha='center')
        
        # 设置坐标轴
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('运动学仿真动画 - 导弹、无人机和烟幕干扰弹')
        
        # 设置坐标轴范围
        self.ax.set_xlim(-2000, 21000)
        self.ax.set_ylim(-500, 500)
        self.ax.set_zlim(0, 2500)
        
        # 添加网格和图例
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', bbox_to_anchor=(0, 1))
        
        # 添加时间文本
        self.time_text = self.ax.text2D(0.02, 0.98, '', transform=self.ax.transAxes, 
                                       fontsize=12, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def generate_cylinder_points(self, n_points=1000):
        """生成圆柱体上下底面的离散点"""
        # 圆柱体参数
        center_bottom = np.array([0, 200, 0])  # 下底面圆心
        center_top = np.array([0, 200, 10])    # 上底面圆心
        radius = 7.0  # 半径
        
        # 生成角度
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        # 下底面点
        bottom_points = []
        for angle in angles:
            x = center_bottom[0] + radius * np.cos(angle)
            y = center_bottom[1] + radius * np.sin(angle)
            z = center_bottom[2]
            bottom_points.append([x, y, z])
        
        # 上底面点
        top_points = []
        for angle in angles:
            x = center_top[0] + radius * np.cos(angle)
            y = center_top[1] + radius * np.sin(angle)
            z = center_top[2]
            top_points.append([x, y, z])
        
        return np.array(bottom_points), np.array(top_points)
    
    def point_to_line_distance(self, point, line_start, line_end):
        """计算点到直线的距离"""
        # 直线方向向量
        line_vec = line_end - line_start
        # 点到直线起点的向量
        point_vec = point - line_start
        
        # 如果直线长度为0，返回点到起点的距离
        line_length_squared = np.dot(line_vec, line_vec)
        if line_length_squared == 0:
            return np.linalg.norm(point_vec)
        
        # 计算投影参数
        t = np.dot(point_vec, line_vec) / line_length_squared
        
        # 找到直线上最近的点
        if t < 0:
            closest_point = line_start
        elif t > 1:
            closest_point = line_end
        else:
            closest_point = line_start + t * line_vec
        
        # 返回距离
        return np.linalg.norm(point - closest_point)
    
    def on_key_press(self, event):
        """处理键盘事件，实现暂停/继续"""
        if event.key == ' ':
            self.is_paused = not self.is_paused
            if self.is_paused:
                print("动画已暂停。按空格键继续。")
            else:
                print("动画已恢复。")
    
    def check_line_sphere_intersection(self, line_start, line_end, sphere_center, sphere_radius):
        """检查直线是否与球体相交"""
        distance = self.point_to_line_distance(sphere_center, line_start, line_end)
        return distance <= sphere_radius
    
    def check_occlusion(self):
        """检查导弹是否被烟幕云团遮挡（复杂逻辑）"""
        if self.cloud is None:
            return False
        
        missile_pos = self.missile_m1.pos
        
        # 首先检查导弹是否在烟幕球体内
        distance_to_cloud = np.linalg.norm(missile_pos - self.cloud.pos)
        if distance_to_cloud <= self.cloud.radius:
            return True  # 导弹在球体内，直接被遮挡
        
        # 导弹在球体外，检查视线遮挡
        # 生成圆柱体上下底面的离散点
        bottom_points, top_points = self.generate_cylinder_points(10000)
        
        # 合并所有目标点
        all_target_points = np.vstack([bottom_points, top_points])
        
        # 检查每条视线是否被烟幕球体遮挡
        occluded_lines = 0
        total_lines = len(all_target_points)
        
        for target_point in all_target_points:
            # 从导弹到目标点的视线
            if not self.check_line_sphere_intersection(missile_pos, target_point, 
                                                 self.cloud.pos, self.cloud.radius):
                return False
        
        # 如果所有视线都被遮挡，则判定为遮挡
        return True  
    
    def update_occlusion_logic(self):
        """更新遮挡逻辑"""
        if self.current_time >= T_DETONATE :
            currently_occluded = self.check_occlusion()
            
            if currently_occluded and not self.is_occluded:
                # 开始遮挡
                self.is_occluded = True
                self.occlusion_start_time = self.current_time
                print(f"时间 {self.current_time:.5f}s: 导弹开始遮挡")
                
            elif not currently_occluded and self.is_occluded:
                # 结束遮挡
                self.is_occluded = False
                self.occlusion_end_time = self.current_time
                occlusion_duration = self.occlusion_end_time - self.occlusion_start_time
                self.occlusion_periods.append((self.occlusion_start_time, self.occlusion_end_time))
                self.total_occlusion_time += occlusion_duration
                print(f"时间 {self.current_time:.5f}s: 导弹结束遮挡 (持续时间: {occlusion_duration:.4f}s)")

    def update_simulation(self, dt):
        """更新仿真状态，使用动态步长精确处理事件"""
        # 检查是否超过仿真时间或仿真已停止
        if self.current_time >= SIM_END_TIME:
            self.simulation_running = False
            return False

        if not self.simulation_running:
            return False
            
        # --- 精确事件处理 ---
        # 1. 确定下一个最近的事件时间
        upcoming_events = []
        if self.Bomb is None and self.cloud is None:
            upcoming_events.append(T_DROP)
        if self.cloud is None and self.Bomb is not None:
            upcoming_events.append(T_DETONATE)
        
        # 2. 动态调整当前步长，确保不会错过事件
        effective_dt = dt
        if upcoming_events:
            next_event_time = min(upcoming_events)
            time_to_next_event = next_event_time - self.current_time
            if time_to_next_event < dt:
                effective_dt = time_to_next_event
        # --- 事件处理结束 ---

        # 使用调整后的有效步长来更新所有物体
        self.missile_m1.update(effective_dt)
        self.drone_fy1.update(effective_dt)
        
        # 检查烟幕弹投放事件 (使用精确时间比较)
        if self.current_time >= T_DROP and self.Bomb is None and self.cloud is None:
            self.Bomb = SmokeBomb(self.drone_fy1.pos.copy(), self.drone_fy1.velocity.copy())
            print(f"时间 {self.current_time:.4f}s: 烟幕弹投放 (精确时刻)")

        # 更新烟幕弹（如果已投放）
        if self.Bomb:
            self.Bomb.update(effective_dt)
        
        # 检查烟幕弹起爆事件 (使用精确时间比较)
        if self.current_time >= T_DETONATE and self.cloud is None and self.Bomb is not None:
            self.detonation_point = self.Bomb.pos.copy()
            self.cloud = SmokeCloud(self.detonation_point)
            self.Bomb = None  # 弹体消失，变为云团
            print(f"时间 {self.current_time:.4f}s: 烟幕弹起爆，形成烟幕云团 (精确时刻)")

            # 起爆后，更改时间步长以提高精度
            if self.dt > 0.01:
                print(f"时间 {self.current_time:.4f}s: 仿真精度提高，时间步长从 {self.dt}s 更改为 0.0001s")
                self.dt = 0.0001

        # 更新烟幕云团（如果已形成）
        if self.cloud:
            self.cloud.update(effective_dt)
        
        # 更新遮挡逻辑
        self.update_occlusion_logic()
        
        # 使用有效步长推进总时间
        self.current_time += effective_dt
        return True
        
    def animate(self, frame):
        """动画更新函数"""
        # 如果暂停，则不更新任何内容
        if self.is_paused:
            return (self.missile_line, self.missile_point, self.drone_line, self.drone_point,
                    self.Bomb_line, self.Bomb_point, self.cloud_line, self.cloud_point, 
                    self.time_text)

        # 更新帧计数
        self.frame_count = frame
        
        # 更新仿真，如果返回False则停止
        if not self.update_simulation(self.dt):
            return (self.missile_line, self.missile_point, self.drone_line, self.drone_point,
                    self.Bomb_line, self.Bomb_point, self.cloud_line, self.cloud_point, 
                    self.time_text)
        
        # 更新导弹
        missile_traj = np.array(self.missile_m1.trajectory)
        if len(missile_traj) > 0:
            self.missile_line.set_data_3d(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2])
            self.missile_point.set_data_3d([self.missile_m1.pos[0]], [self.missile_m1.pos[1]], [self.missile_m1.pos[2]])
        
        # 更新无人机
        drone_traj = np.array(self.drone_fy1.trajectory)
        if len(drone_traj) > 0:
            self.drone_line.set_data_3d(drone_traj[:, 0], drone_traj[:, 1], drone_traj[:, 2])
            self.drone_point.set_data_3d([self.drone_fy1.pos[0]], [self.drone_fy1.pos[1]], [self.drone_fy1.pos[2]])
        
        # 更新烟幕弹
        if self.Bomb:
            proj_traj = np.array(self.Bomb.trajectory)
            if len(proj_traj) > 0:
                self.Bomb_line.set_data_3d(proj_traj[:, 0], proj_traj[:, 1], proj_traj[:, 2])
                self.Bomb_point.set_data_3d([self.Bomb.pos[0]], [self.Bomb.pos[1]], [self.Bomb.pos[2]])
        else:
            # 隐藏烟幕弹
            self.Bomb_point.set_data_3d([], [], [])
        
        # 更新烟幕云团
        if self.cloud:
            cloud_traj = np.array(self.cloud.trajectory)
            if len(cloud_traj) > 0:
                self.cloud_line.set_data_3d(cloud_traj[:, 0], cloud_traj[:, 1], cloud_traj[:, 2])
                self.cloud_point.set_data_3d([self.cloud.pos[0]], [self.cloud.pos[1]], [self.cloud.pos[2]])
                
                # 绘制烟幕球体（每10帧更新一次以提高性能）
                if frame % 10 == 0:
                    # 清除之前的球体
                    for artist in self.ax.collections:
                        if hasattr(artist, '_smoke_sphere'):
                            artist.remove()
                    
                    # 绘制新的烟幕球体
                    try:
                        draw_sphere(self.ax, self.cloud.pos, self.cloud.radius, 'gray', 0.2)
                        if hasattr(self.ax.collections[-1], 'set_alpha'):
                            self.ax.collections[-1]._smoke_sphere = True
                    except:
                        pass  # 如果绘制失败，跳过
        else:
            # 隐藏烟幕云团
            self.cloud_point.set_data_3d([], [], [])
        
        # 更新时间显示
        status_text = f'时间: {self.current_time:.4f}s\n'
        if self.current_time < T_DROP:
            status_text += '状态: 导弹和无人机飞行中'
        elif self.current_time < T_DETONATE:
            status_text += '状态: 烟幕弹投放，斜抛运动中'
        elif self.current_time < T_EFFECT_END:
            if self.is_occluded:
                status_text += '状态: 导弹被遮挡中'
            else:
                status_text += '状态: 导弹未被遮挡'
            # 添加遮挡比例信息
            if hasattr(self, 'occlusion_ratio'):
                status_text += f'\n遮挡比例: {self.occlusion_ratio*100:.4f}%'
            status_text += f'\n总遮挡时间: {self.total_occlusion_time:.4f}s'
        else:
            status_text += '状态: 仿真结束'
            status_text += f'\n总遮挡时间: {self.total_occlusion_time:.4f}s'
            
        self.time_text.set_text(status_text)
        
        # 强制刷新显示（实时显示动画）
        
        plt.draw()
        plt.pause(0.0001)  # 很短的暂停让界面有时间更新
        
        return (self.missile_line, self.missile_point, self.drone_line, self.drone_point,
                self.Bomb_line, self.Bomb_point, self.cloud_line, self.cloud_point, 
                self.time_text)

    def on_key_press(self, event):
        """键盘事件处理器"""
        if event.key == 'p':
            self.is_paused = not self.is_paused
            if self.is_paused:
                print("动画已暂停")
            else:
                print("动画已恢复")
        elif event.key == 'r':
            self.current_time = 0
            self.missile_m1.pos = M1_INITIAL_POS
            self.drone_fy1.pos = FY1_INITIAL_POS
            self.Bomb = None
            self.cloud = None
            self.detonation_point = None
            self.missile_m1.trajectory = [M1_INITIAL_POS]
            self.drone_fy1.trajectory = [FY1_INITIAL_POS]
            self.missile_m1.velocity = np.array([0, 0, 0]) # 停止运动
            self.drone_fy1.velocity = np.array([0, 0, 0])
            print("仿真已重置")
        elif event.key == 'q':
            self.simulation_running = False
            print("仿真已停止")

# --- 5. 运行动画 ---

if __name__ == "__main__":
    print("开始运动学仿真动画...")
    print("="*50)
    print("仿真参数:")
    print(f"- 导弹初始位置: {M1_INITIAL_POS}")
    print(f"- 导弹速度: {M1_SPEED} m/s")
    print(f"- 无人机初始位置: {FY1_INITIAL_POS}")
    print(f"- 无人机速度: {FY1_SPEED} m/s")
    print(f"- 烟幕弹投放时间: {T_DROP}s")
    print(f"- 烟幕弹起爆时间: {T_DETONATE}s")
    print(f"- 真目标位置: {TRUE_TARGET_POS} (圆柱体: 半径{CYLINDER_RADIUS}m, 高{CYLINDER_HEIGHT}m)")
    print(f"- 假目标位置: {FALSE_TARGET_POS} (导弹瞄准点)")
    print("="*50)
    print("提示: 在动画窗口中按 空格键 可以暂停/继续动画。")
    
    # 创建动画实例
    anim_sim = KinematicAnimation()
    
    # 创建动画 - 使用合理的播放速度
    total_frames = int(SIM_END_TIME / DT)  # 计算实际需要的帧数
    
    print(f"总仿真帧数: {total_frames}")
    print(f"动画播放间隔: 20ms 每帧 (加速播放)")
    
    # 立即显示图形窗口
    print("正在初始化动画窗口...")
    plt.draw()  # 绘制初始状态
    plt.show(block=False)  # 非阻塞显示，立即显示窗口
    plt.pause(1.0)  # 让窗口有时间完全初始化并显示
    
    # 使用较快的播放速度来避免动画过长
    ani = animation.FuncAnimation(anim_sim.fig, anim_sim.animate, 
                                 interval=1, blit=False, repeat=False)
    
    # 保存动画为GIF（可选）
    save_gif = True  # 可以设置为False跳过保存
    if save_gif:
        try:
            print("正在保存GIF文件（可能需要一些时间）...")
            ani.save('t1_highjd.gif', writer='pillow', fps=10)
            print("动画已保存为 'kinematic_animation.gif'")
        except:
            print("无法保存GIF，但动画正在窗口中播放")
    
    # 显示动画
    plt.show()
    
    # 输出最终遮挡统计
    print("\n" + "="*50)
    print("遮挡统计结果:")
    print(f"总遮挡时间: {anim_sim.total_occlusion_time:.2f}秒")
    print(f"遮挡次数: {len(anim_sim.occlusion_periods)}次")
    if anim_sim.occlusion_periods:
        print("遮挡时间段:")
        for i, (start, end) in enumerate(anim_sim.occlusion_periods):
            duration = end - start
            print(f"  第{i+1}次: {start:.4f}s - {end:.4f}s (持续 {duration:.4f}s)")
    else:
        print("导弹未被烟幕云团遮挡")
    print("="*50)
    
    print("动画完成！")
