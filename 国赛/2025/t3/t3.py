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
G = 9.80665
CLOUD_RADIUS = 10.0
SINK_VELOCITY = np.array([0.0, 0.0, -3.0]) # 烟幕云团下沉速度
SIM_DT = 0.01 # 仿真时间步长
CLOUD_DURATION = 20.0 # 每个烟幕云的持续时间

# --- 2. 辅助和核心计算函数 ---

# 预先计算导弹速度向量
m1_direction = M1_TARGET_POS - M1_INITIAL_POS
M1_VELOCITY = m1_direction / np.linalg.norm(m1_direction) * M1_SPEED

def get_missile_pos(t):
    """计算t时刻导弹的位置"""
    return M1_INITIAL_POS + M1_VELOCITY * t

def generate_cylinder_points(n_points=100):
    """生成圆柱体轮廓上的离散点"""
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
    """
    line_vecs = lines_ends - lines_start
    point_vec = cloud_point - lines_start
    line_lens_sq = np.sum(line_vecs**2, axis=1)
    line_lens_sq[line_lens_sq == 0] = 1e-9
    t_numerator = np.sum(point_vec * line_vecs, axis=1)
    t = np.clip(t_numerator / line_lens_sq, 0, 1)
    closest_points = lines_start + np.expand_dims(t, axis=1) * line_vecs
    return np.sum((closest_points - cloud_point)**2, axis=1)

def get_occluded_lines_mask_by_single_cloud(missile_pos, cloud_pos):
    """
    检查单个烟幕云，并返回一个布尔数组，指示它遮挡了哪些视线。
    """
    # 如果导弹在烟幕球体内，则所有视线都被遮挡
    if np.sum((missile_pos - cloud_pos)**2) <= CLOUD_RADIUS_SQ:
        return np.ones(len(TARGET_POINTS), dtype=bool)

    # 向量化检查：一次性计算烟幕球心到所有视线(线段)的距离
    dist_sq_all_lines = vectorized_cloud_to_lines_distance_sq(cloud_pos, missile_pos, TARGET_POINTS)
    
    # 条件a: 哪些视线被烟幕球的宽度覆盖
    mask_width = dist_sq_all_lines <= CLOUD_RADIUS_SQ
    
    # 条件b: 对于哪些视线，烟幕处于导弹和目标之间
    dist_sq_cloud_to_targets = np.sum((TARGET_POINTS - cloud_pos)**2, axis=1)
    dist_sq_missile_to_targets = np.sum((TARGET_POINTS - missile_pos)**2, axis=1)
    mask_between = dist_sq_cloud_to_targets - dist_sq_all_lines < dist_sq_missile_to_targets
    
    # 一条视线被遮挡，必须同时满足以上两个条件
    return np.logical_and(mask_width, mask_between)

# --- 3. 目标函数 ---

def calculate_occlusion_time_single(params):
    """
    为单个粒子（一组参数）计算总遮蔽时间。
    参数包含8个变量，对应3枚烟幕弹。
    """
    a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params
    
    # --- 新增：约束条件检查 ---
    # 如果粒子不满足约束，则返回一个巨大的惩罚值 (pyswarms是最小化，所以返回正的大数)
    # 约束1: 投放时间间隔至少1秒
    if t_take2 < t_take1 + 1 or t_take3 < t_take2 + 1:
        return 1000.0 
    
    # 约束2: 下落时间递增
    if not (t_fall3 > t_fall2 > t_fall1):
        return 1000.0
    
    t_takes = np.array([t_take1, t_take2, t_take3])
    t_falls = np.array([t_fall1, t_fall2, t_fall3])
    
    drone_velocity = np.array([v * np.cos(a), v * np.sin(a), 0.0])
    
    # 计算3枚烟幕弹的投放点、引爆点和引爆时间
    T_DROPS = t_takes
    T_DETONATES = T_DROPS + t_falls
    
    drone_pos_at_drops = FY1_INITIAL_POS + drone_velocity * T_DROPS[:, np.newaxis]
    
    detonation_points = drone_pos_at_drops + drone_velocity * t_falls[:, np.newaxis]
    detonation_points[:, 2] -= 0.5 * G * t_falls**2
    
    # 过滤掉在地面以下引爆的无效烟幕弹
    valid_detonations = detonation_points[:, 2] >= 0
    if not np.any(valid_detonations):
        return 0
        
    detonation_points = detonation_points[valid_detonations]
    T_DETONATES = T_DETONATES[valid_detonations]
    
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
    
    # 设置仿真时间范围
    sim_start_time = np.min(T_DETONATES)
    sim_end_time = np.max(T_EFFECT_ENDS)
    time_steps = np.arange(sim_start_time, sim_end_time, SIM_DT)
    
    total_occlusion_time = 0.0
    
    for t in time_steps:
        missile_pos = get_missile_pos(t)
        
        if missile_pos[0] < 0: # 导弹已飞过目标
            break
            
        # 确定在当前时刻 t，哪些烟幕云是激活的
        active_mask = (t >= T_DETONATES) & (t < T_EFFECT_ENDS)
        if not np.any(active_mask):
            continue
            
        active_detonation_points = detonation_points[active_mask]
        active_T_DETONATES = T_DETONATES[active_mask]
        
        # 计算所有激活烟幕云的当前中心位置
        time_since_detonation = t - active_T_DETONATES
        cloud_positions = active_detonation_points + SINK_VELOCITY * time_since_detonation[:, np.newaxis]
        
        # 正确的协同遮挡逻辑：
        # 1. 初始化一个记录所有视线遮挡状态的布尔数组
        occluded_lines_mask = np.zeros(len(TARGET_POINTS), dtype=bool)
        
        # 2. 遍历每一个烟幕云，更新总的遮挡状态
        for cloud_pos in cloud_positions:
            # 获取当前单个烟幕云的遮挡情况
            current_cloud_mask = get_occluded_lines_mask_by_single_cloud(missile_pos, cloud_pos)
            # 使用逻辑或，将当前烟幕云的遮挡效果累加到总状态中
            occluded_lines_mask = np.logical_or(occluded_lines_mask, current_cloud_mask)
        
        # 3. 如果所有视线都被遮挡 (每个视线至少被一个云遮挡)，则判定为完全遮挡
        is_occluded_at_t = np.all(occluded_lines_mask)
        
        if is_occluded_at_t:
             total_occlusion_time += SIM_DT
             
    return -total_occlusion_time

def calculate_occlusion_time_parallel(swarm):
    """
    为 pyswarms 准备的并行化目标函数。
    """
    particles = [particle for particle in swarm]
    with multiprocessing.Pool() as pool:
        results = pool.map(calculate_occlusion_time_single, particles)
    return np.array(results)

# --- 4. 优化过程 ---

if __name__ == "__main__":
    # 定义8个优化变量的边界: [a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3]
    bounds_low = [np.pi-0.0113, 70, 0, 0, 0, 0, 0, 0]
    bounds_high = [np.pi, 140, 1, 5, 10, 10, 20, 20]
    bounds = (np.array(bounds_low), np.array(bounds_high))
    
    print("开始为3枚烟幕弹进行多变量优化...")
    print("请确保您已安装 'pyswarms' 库 (pip install pyswarms)")
    print(f"变量顺序: (飞行角度, 飞行速度, 投放时间1, 下落时间1, 投放时间2, ...)")
    print("优化算法: 并行粒子群优化 (Parallel PSO via pyswarms)")
    print("计算将动用所有CPU核心，请稍候...")
    print("="*60)
    
    start_time = time.time()

    # --- 使用 pyswarms 并行PSO算法 ---
    # 1. 设置超参数: c1(认知), c2(社会), w(惯性)
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

    # 2. 创建优化器实例，维度增加到8
    optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=8, options=options, bounds=bounds)

    # 3. 运行优化器
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
    print("-" * 30)
    print(f"  - 烟幕弹 1: 投放时间 = {best_params[2]:.4f} s, 下落时间 = {best_params[3]:.4f} s")
    print(f"  - 烟幕弹 2: 投放时间 = {best_params[4]:.4f} s, 下落时间 = {best_params[5]:.4f} s")
    print(f"  - 烟幕弹 3: 投放时间 = {best_params[6]:.4f} s, 下落时间 = {best_params[7]:.4f} s")
    print("\n" + "-"*30)
    print(f"  => 最大遮蔽时间: {max_occlusion_time:.4f} 秒")
    print("-" * 60)
