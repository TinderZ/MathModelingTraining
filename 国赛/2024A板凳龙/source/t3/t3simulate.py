import numpy as np
from scipy.optimize import fsolve, root_scalar
import math
import time as timer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DragonRectangleSimulation:
    """
    修改后的仿真类，用于接受一个可变的螺距(pitch)。
    大部分代码逻辑与 t2simulate.py 中的相同。
    """
    def __init__(self, pitch):
        # --- 1. 定义系统常量和参数 ---
        self.pitch = pitch
        self.spiral_coeff = self.pitch / (2.0 * np.pi) # a = p / (2*pi)

        self.num_segments = 100
        self.first_segment_length = 286.0
        self.segment_length = 165.0
        self.speed = 100.0

        self.dt = 0.1
        self.time = 0.0

        self.collision_detected = False
        self.collision_time = -1.0
        
        # --- 2. 矩形板凳龙的几何参数 ---
        self.dragon_head_length = 341.0
        self.dragon_body_length = 220.0
        self.board_width = 30.0
        
        self.head_rect_length = self.dragon_head_length
        self.head_rect_width = self.board_width
        self.body_rect_length = self.dragon_body_length
        self.body_rect_width = self.board_width

        # --- 3. 创建杆长数组 ---
        self.lengths = np.full(self.num_segments, self.segment_length)
        self.lengths[0] = self.first_segment_length

        # --- 4. 初始化仿真状态 ---
        self.thetas = np.zeros(self.num_segments + 1)
        self.thetas[0] = 24.3 * np.pi
        
        self.initialize_thetas()

    def get_spiral_xy(self, theta):
        r = self.spiral_coeff * theta
        return r * np.cos(theta), r * np.sin(theta)

    def arc_length_from_origin(self, theta):
        a = self.spiral_coeff
        return 0.5 * a * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))
        
    @staticmethod
    def distance_equation(theta2, theta1, L, a):
        r1 = a * theta1
        r2 = a * theta2
        dx = r1 * np.cos(theta1) - r2 * np.cos(theta2)
        dy = r1 * np.sin(theta1) - r2 * np.sin(theta2)
        return dx*dx + dy*dy - L*L



    def initialize_thetas(self):
        for i in range(self.num_segments):
            theta_prev = self.thetas[i]
            L_current = self.lengths[i]
            if theta_prev > 1:
                guess = theta_prev + L_current / (self.spiral_coeff * theta_prev)
            else:
                guess = theta_prev + 0.1
            
            # 使用 try-except 块来捕获 fsolve 可能的失败
            try:
                theta_next, = fsolve(self.distance_equation, guess,
                                     args=(theta_prev, L_current, self.spiral_coeff),
                                     xtol=1e-8) # 增加容忍度以提高稳定性
                self.thetas[i+1] = theta_next
            except (RuntimeError, ValueError) as e:
                # 如果求解失败，抛出一个异常，由调用者处理
                raise RuntimeError(f"初始化节点 {i+1} 失败: {e}") from e


    def get_rectangle_corners(self, center_x, center_y, angle, segment_index):
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

    def project_polygon(self, axis, polygon):
        projections = np.dot(polygon, axis)
        return np.min(projections), np.max(projections)

    def check_collision_sat(self, poly1, poly2):
        polygons = [poly1, poly2]
        for polygon in polygons:
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                edge = p2 - p1
                axis = np.array([-edge[1], edge[0]])
                min1, max1 = self.project_polygon(axis, poly1)
                min2, max2 = self.project_polygon(axis, poly2)
                if max1 < min2 or max2 < min1:
                    return False
        return True

    def check_for_collisions(self):
        all_rects = self.get_all_rectangles()
        rect_inner_0 = all_rects[0]
        rect_inner_1 = all_rects[1]
        
        theta1 = self.thetas[0]
        theta2 = self.thetas[2]
        k1 = np.searchsorted(self.thetas, theta1 + 2 * np.pi)
        k2 = np.searchsorted(self.thetas, theta2 + 2 * np.pi)
        
        start_idx = max(2, k1 - 2)
        end_idx = min(self.num_segments - 1, k2 + 2)

        for i in range(start_idx, end_idx + 1):
            rect_outer = all_rects[i]
            if self.check_collision_sat(rect_inner_0, rect_outer):
                return True
            if self.check_collision_sat(rect_inner_1, rect_outer):
                return True
        return False

    def get_all_rectangles(self):
        rectangles = []
        for i in range(self.num_segments):
            x1, y1 = self.get_spiral_xy(self.thetas[i])
            x2, y2 = self.get_spiral_xy(self.thetas[i+1])
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            angle = np.arctan2(y2 - y1, x2 - x1)
            corners = self.get_rectangle_corners(center_x, center_y, angle, i)
            rectangles.append(corners)
        return rectangles

    def step(self):
        # if self.time < 400.0: self.dt = 1.0
        # elif self.time < 411.0: self.dt = 0.1
        # elif self.time < 412.0: self.dt = 0.01
        # else: self.dt = 0.0001

        if self.collision_detected: return

        theta_old = self.thetas[0]
        s_old = self.arc_length_from_origin(theta_old)
        delta_s = self.speed * self.dt
        s_new_target = s_old - delta_s
        
        if s_new_target <= 0:
            self.thetas[0] = 0
        else:
            def find_theta_eq(theta_new):
                return self.arc_length_from_origin(theta_new) - s_new_target
            sol = root_scalar(find_theta_eq, bracket=[0, theta_old], method='brentq', xtol=1e-15)
            if sol.converged: self.thetas[0] = sol.root
            else: return

        for i in range(self.num_segments):
            theta_prev = self.thetas[i]
            L_current = self.lengths[i]
            initial_guess = self.thetas[i+1]
            theta_next, = fsolve(self.distance_equation, initial_guess, 
                               args=(theta_prev, L_current, self.spiral_coeff), xtol=1e-8)
            self.thetas[i+1] = theta_next
            
        self.time += self.dt

        if self.check_for_collisions():
            self.collision_detected = True
            self.collision_time = self.time

    def plot_spiral_background(self, ax, theta_range_max):
        """绘制螺旋线背景"""
        theta_spiral_bg = np.linspace(0, theta_range_max, 2000)
        x_spiral_bg, y_spiral_bg = self.get_spiral_xy(theta_spiral_bg)
        ax.plot(x_spiral_bg, y_spiral_bg, '--', color='gray', linewidth=0.8, label='螺旋轨迹')


def run_animation_for_pitch(pitch, target_theta, verbose=False):
    """
    为单个螺距p运行一次完整的可视化仿真。
    此版本会在达到目标或碰撞后暂停动画，等待用户手动关闭窗口。
    返回: (bool) 是否在到达目标theta前发生碰撞。
    """
    if verbose:
        print(f"--- 正在可视化测试螺距 p = {pitch:.4f} ---")

    try:
        sim = DragonRectangleSimulation(pitch=pitch)
    except RuntimeError as e:
        if verbose:
            print(f"仿真失败 (p={pitch:.4f}): {e}。假定为碰撞。")
        return True

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title(f'正在测试螺距 p = {pitch:.4f}')

    line, = ax.plot([], [], 'o-', lw=1, markersize=2, color='red', alpha=0.7, label='节点连接')
    rectangles_collection = PatchCollection([], facecolors='lightblue', edgecolors='blue', alpha=0.6)
    ax.add_collection(rectangles_collection)
    title = ax.set_title("初始化...")
    ax.set_aspect('equal')
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.grid(True)

    def init():
        sim.plot_spiral_background(ax, sim.thetas[0] * 1.5)
        max_r = sim.spiral_coeff * sim.thetas[0] * 1.5
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.legend()
        return line, rectangles_collection, title

    def animate(frame):
        # 如果没有碰撞且未达到目标，则推进仿真
        if not sim.collision_detected and sim.thetas[0] > target_theta:
            sim.step()
        
        x_coords, y_coords = sim.get_spiral_xy(sim.thetas)
        line.set_data(x_coords, y_coords)
        
        patches = [Polygon(corners, closed=True) for corners in sim.get_all_rectangles()]
        rectangles_collection.set_paths(patches)
        
        status_text = "运行中..."
        if sim.collision_detected:
            status_text = f"碰撞! 时间 t={sim.collision_time:.2f}s. 请关闭窗口继续。"
        elif sim.thetas[0] <= target_theta:
            status_text = f"已到达目标角度. 请关闭窗口继续。"

        title.set_text(f"p={pitch:.4f} | {status_text} | 龙头角度:{sim.thetas[0]:.4f}rad")
        return line, rectangles_collection, title

    # repeat=True 确保动画循环，暂停逻辑在 animate 函数内部处理
    ani = FuncAnimation(fig, animate, init_func=init, blit=False, interval=1, repeat=True, cache_frame_data=False)
    
    plt.tight_layout()
    # 使用阻塞式 show()，程序会在此暂停直到窗口被关闭
    plt.show()

    # 窗口关闭后，根据最终状态返回结果
    if sim.collision_detected:
        if verbose:
            print(f"碰撞! 时间 t={sim.collision_time:.4f}s, 龙头角度={sim.thetas[0]:.4f} rad.")
        return True
    else:
        if verbose:
            if sim.thetas[0] <= target_theta:
                print(f"安全! 到达目标角度 {sim.thetas[0]:.4f} rad，未发生碰撞。")
            else:
                print(f"窗口被手动关闭! 仿真停止于龙头角度 {sim.thetas[0]:.4f} rad。")
        return False


def does_collide_early(pitch, target_theta, verbose=False):
    """
    通过运行可视化仿真来检查是否提前碰撞。
    """
    return run_animation_for_pitch(pitch, target_theta, verbose)


def find_minimum_pitch():
    """
    使用二分搜索寻找满足条件的最小螺距p。
    """
    low_p = 45.0   # 搜索下界 (cm)，基于板凳宽度
    high_p = 46.0 # 搜索上界 (cm)，一个足够大的初始值
    tolerance = 0.0001 # 搜索精度 (cm)
    TARGET_THETA =  900*np.pi/((low_p + high_p) / 2.0)

    print("="*60)
    print("开始二分搜索寻找最小安全螺距 p...")
    print(f"搜索范围: [{low_p}, {high_p}], 精度: {tolerance}")
    print(f"安全条件: 在龙头角度 > {TARGET_THETA:.4f} rad 时不发生碰撞")
    print("="*60)



    iterations = 0
    max_iterations = 100
    
    start_time = timer.time()

    while high_p - low_p > tolerance and iterations < max_iterations:
        iterations += 1
        mid_p = (low_p + high_p) / 2.0
        
        print(f"\n[迭代 {iterations}] 范围 [{low_p:.4f}, {high_p:.4f}]")
        
        collided = does_collide_early(mid_p, (900*np.pi/mid_p), verbose=True)
        
        if collided:
            # mid_p 不安全，需要更大的螺距
            print(f"结论: p={mid_p:.4f} 不安全。更新下界: low_p = {mid_p:.4f}")
            low_p = mid_p
        else:
            # mid_p 是安全的，尝试更小的螺距
            print(f"结论: p={mid_p:.4f} 安全。更新上界: high_p = {mid_p:.4f}")
            high_p = mid_p
    
    end_time = timer.time()
    
    print("\n" + "="*60)
    print("搜索完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"找到的最小安全螺距 p ≈ {high_p:.4f} cm")
    print("="*60)
    
    return high_p

if __name__ == "__main__":
    find_minimum_pitch()
