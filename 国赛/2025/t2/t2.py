import numpy as np
import pyswarms as ps
import time
import multiprocessing

# --- 1. 常量和参数定义 ---

# 导弹M1参数
M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
M1_TARGET_POS = np.array([0.0, 0.0, 0.0])
M1_SPEED = 300.0

# 无人机FY1参数
FY1_INITIAL_POS = np.array([17800.0, 0.0, 1800.0])

# 目标圆柱体参数
CYLINDER_CENTER_BOTTOM = np.array([0.0, 200.0, 0.0])
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0

# 物理和仿真参数
G = 9.80665  # 重力加速度
CLOUD_RADIUS = 10.0
SINK_VELOCITY = np.array([0.0, 0.0, -3.0]) # 烟幕云团下沉速度
SIM_DT = 0.01 # 仿真时间步长

# --- 2. 辅助和核心计算函数 ---

# 预先计算导弹速度向量，因为它是不变的
m1_direction = M1_TARGET_POS - M1_INITIAL_POS
M1_VELOCITY = m1_direction / np.linalg.norm(m1_direction) * M1_SPEED

def get_missile_pos(t):
    """计算t时刻导弹的位置"""
    return M1_INITIAL_POS + M1_VELOCITY * t

def generate_cylinder_points(n_points=100):
    """
    生成圆柱体轮廓上的离散点。
    """
    center_bottom = CYLINDER_CENTER_BOTTOM
    center_top = center_bottom + np.array([0, 0, CYLINDER_HEIGHT])
    radius = CYLINDER_RADIUS
    
    angles = np.linspace(0, 2 * np.pi, n_points // 2, endpoint=False)
    bottom_x = center_bottom[0] + radius * np.cos(angles)
    bottom_y = center_bottom[1] + radius * np.sin(angles)
    bottom_z = np.full_like(bottom_x, center_bottom[2])
    
    top_x = center_top[0] + radius * np.cos(angles)
    top_y = center_top[1] + radius * np.sin(angles)
    top_z = np.full_like(top_x, center_top[2])

    points_x = np.concatenate([bottom_x, top_x])
    points_y = np.concatenate([bottom_y, top_y])
    points_z = np.concatenate([bottom_z, top_z])
    
    return np.vstack([points_x, points_y, points_z]).T

TARGET_POINTS = generate_cylinder_points(100)
CLOUD_RADIUS_SQ = CLOUD_RADIUS**2

def vectorized_cloud_to_lines_distance_sq(cloud_point, lines_start, lines_ends):
    """
    向量化版本：同时计算一个点 (烟幕中心) 到多条线段 (视线) 的距离的平方。
    cloud_point: (3,) a single point
    lines_start: (3,) the common start point for all lines (missile_pos)
    lines_ends: (N, 3) the end points for all lines (target_points)
    """
    # line_vecs: 存储所有视线向量, shape: (N, 3)
    line_vecs = lines_ends - lines_start
    
    # point_vec: 从视线起点(导弹)指向烟幕中心的向量, shape: (3,)
    point_vec = cloud_point - lines_start
    
    # line_lens_sq: 每条视线长度的平方, shape: (N,)
    line_lens_sq = np.sum(line_vecs**2, axis=1)
    
    # 防止除以零
    line_lens_sq[line_lens_sq == 0] = 1e-9
    
    # t_numerator: 烟幕中心向量在每条视线向量上的投影长度
    # (3,) * (N, 3) -> (N, 3), then sum over axis 1 -> (N,)
    t_numerator = np.sum(point_vec * line_vecs, axis=1)

    # t: 投影比例, shape: (N,)
    t = t_numerator / line_lens_sq
    t = np.clip(t, 0, 1)

    # closest_points: 烟幕中心在每条视线上的最近点, shape: (N, 3)
    # (3,) + (N, 1) * (N, 3) -> (N, 3)
    closest_points = lines_start + np.expand_dims(t, axis=1) * line_vecs
    
    # 返回烟幕中心到每个最近点的距离的平方, shape: (N,)
    return np.sum((closest_points - cloud_point)**2, axis=1)

def check_occlusion(missile_pos, cloud_pos):
    # 1. 检查导弹是否在烟幕球体内
    if np.sum((missile_pos - cloud_pos)**2) <= CLOUD_RADIUS_SQ:
        return True

    # 2. 向量化检查：一次性计算烟幕球心到所有视线(线段)的距离
    dist_sq_all_lines = vectorized_cloud_to_lines_distance_sq(cloud_pos, missile_pos, TARGET_POINTS)
    
    # 条件2a: 检查所有视线是否都被烟幕球的宽度覆盖
    is_occluded_by_width = np.all(dist_sq_all_lines <= CLOUD_RADIUS_SQ)
    if not is_occluded_by_width:
        return False

    # 条件2b: 检查烟幕是否处于导弹和目标之间
    # 通过比较距离的平方来实现，确保烟幕中心的投影点落在导弹和目标点之间
    dist_sq_cloud_to_targets = np.sum((TARGET_POINTS - cloud_pos)**2, axis=1)
    dist_sq_missile_to_targets = np.sum((TARGET_POINTS - missile_pos)**2, axis=1)
    
    is_between = np.all(dist_sq_cloud_to_targets - dist_sq_all_lines < dist_sq_missile_to_targets)
    
    return is_between

# --- 3. 目标函数 ---

def calculate_occlusion_time_single(params):
    a, v, t_take, t_fall = params
    
    T_DROP = t_take
    T_DETONATE = T_DROP + t_fall
    T_EFFECT_END = T_DETONATE + 20

    drone_velocity = np.array([v * np.cos(a), v * np.sin(a), 0.0])
    drone_pos_at_drop = FY1_INITIAL_POS + drone_velocity * T_DROP

    detonation_point = drone_pos_at_drop + drone_velocity * t_fall
    detonation_point[2] -= 0.5 * G * t_fall**2
    
    # --- 新增：引爆点到视线距离约束 ---
    missile_pos_at_detonation = get_missile_pos(T_DETONATE)
    target_center = np.array([0.0, 200.0, 5.0])
    
    vec_AP = detonation_point - missile_pos_at_detonation
    vec_AB = target_center - missile_pos_at_detonation
    
    norm_vec_AB = np.linalg.norm(vec_AB)
    
    # 避免除以零
    if norm_vec_AB < 1e-9:
        norm_vec_AB = 1e-9
    
    distance = np.linalg.norm(np.cross(vec_AP, vec_AB)) / norm_vec_AB
    
    if distance > 60:
        return 102.0 # 返回一个巨大的惩罚值
    
    if detonation_point[2] < 0:
        return 0

    total_occlusion_time = 0.0
    time_steps = np.arange(T_DETONATE, T_EFFECT_END, SIM_DT)
    
    for t in time_steps:
        missile_pos = get_missile_pos(t)
        
        if missile_pos[0] < 0:
            break

        time_since_detonation = t - T_DETONATE
        cloud_pos = detonation_point + SINK_VELOCITY * time_since_detonation

        if check_occlusion(missile_pos, cloud_pos):
             total_occlusion_time += SIM_DT
    
    return -total_occlusion_time

def calculate_occlusion_time_parallel(swarm):
    """
    为 pyswarms 准备的并行化目标函数。
    它接收整个粒子群 (swarm)，并使用 multiprocessing.Pool 将计算
    任务分配给所有可用的CPU核心。
    """
    # 将swarm数组转换为可迭代的列表，供 pool.map 使用
    particles = [particle for particle in swarm]
    # 创建一个进程池，它会自动使用所有可用的CPU核心
    with multiprocessing.Pool() as pool:
        # map函数会将particles列表中的每个元素分配给一个进程去执行
        results = pool.map(calculate_occlusion_time_single, particles)
    return np.array(results)

# --- 4. 优化过程 ---

if __name__ == "__main__":
    # 定义优化变量的边界: [a, v, t_take, t_fall]
    bounds_low = [np.pi-0.26, 70, 0, 0]
    bounds_high = [np.pi, 140, 2, 5]
    bounds = (bounds_low, bounds_high)
    
    print("开始进行多变量优化以寻找最大遮蔽时间...")
    print("请确保您已安装 'pyswarms' 库 (pip install pyswarms)")
    print(f"变量顺序: (飞行角度, 飞行速度, 投放时间, 下落时间)")
    print("优化算法: 并行粒子群优化 (Parallel PSO via pyswarms)")
    print("计算正在进行中，将动用所有CPU核心，请稍候...")
    print("="*60)
    
    start_time = time.time()

    # --- 使用 pyswarms 并行PSO算法 ---
    # 1. 设置超参数: c1(认知), c2(社会), w(惯性)
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

    # 2. 创建优化器实例
    optimizer = ps.single.GlobalBestPSO(n_particles=3000, dimensions=4, options=options, bounds=bounds)

    # 3. 运行优化器，传入我们自定义的并行目标函数
    # iters: 迭代次数
    best_cost, best_params = optimizer.optimize(calculate_occlusion_time_parallel, iters=100, verbose=True)
    
    end_time = time.time()
    
    print("="*60)
    print("优化完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("-" * 60)
    
    max_occlusion_time = -best_cost
    
    print("找到的最佳结果:")
    print(f"  - 飞行方向角 (a): {np.rad2deg(best_params[0]):.4f} 度")
    print(f"  - 飞行速度 (v): {best_params[1]:.4f} m/s")
    print(f"  - 烟幕弹投放时间 (t_take): {best_params[2]:.4f} s")
    print(f"  - 烟幕弹下落时间 (t_fall): {best_params[3]:.4f} s")
    print("\n" + "-"*30)
    print(f"  => 最大遮蔽时间: {max_occlusion_time:.4f} 秒")
    print("-" * 60)
