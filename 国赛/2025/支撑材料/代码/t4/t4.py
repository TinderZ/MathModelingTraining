import numpy as np
import pyswarms as ps
import time
import multiprocessing
import itertools

# --- 1. 常量和参数定义 ---

# 导弹M1参数
M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
M1_TARGET_POS = np.array([0.0, 0.0, 0.0])
M1_SPEED = 300.0

# 无人机初始位置 (FY1, FY2, FY3)
DRONES_INITIAL_POS = np.array([
    [17800.0, 0.0, 1800.0],    # FY1
    [12000.0, 1400.0, 1400.0], # FY2
    [6000.0, -3000.0, 700.0]  # FY3
])

# 目标圆柱体参数
CYLINDER_CENTER_BOTTOM = np.array([0.0, 200.0, 0.0])
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0

# 物理和仿真参数
G = 9.80665
CLOUD_RADIUS = 10.0
SINK_VELOCITY = np.array([0.0, 0.0, -3.0])
SIM_DT = 0.01
CLOUD_DURATION = 20.0

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
    """向量化版本：计算一个点到多条线段的距离的平方"""
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
    """为单个粒子（一组12个参数）计算总遮蔽时间"""
    
    # 将12个参数分解到3架无人机
    params_per_drone = np.array(params).reshape(3, 4)
    
    # --- 新增：约束条件检查 ---
    t_takes = params_per_drone[:, 2]
    t_falls = params_per_drone[:, 3]

    # # 约束: 投放时间均需严格递增
    # if not (t_takes[0] < t_takes[1] < t_takes[2]):
    #     return 1000.0  # 返回一个巨大的惩罚值
    
    drone_velocities = np.zeros((3, 3))
    drone_velocities[:, 0] = params_per_drone[:, 1] * np.cos(params_per_drone[:, 0]) # v * cos(a)
    drone_velocities[:, 1] = params_per_drone[:, 1] * np.sin(params_per_drone[:, 0]) # v * sin(a)
    
    # 计算3枚烟幕弹的投放点、引爆点和引爆时间
    T_DROPS = t_takes
    T_DETONATES = T_DROPS + t_falls
    
    drone_pos_at_drops = DRONES_INITIAL_POS + drone_velocities * T_DROPS[:, np.newaxis]
    
    detonation_points = drone_pos_at_drops + drone_velocities * t_falls[:, np.newaxis]
    detonation_points[:, 2] -= 0.5 * G * t_falls**2
    
    # --- 新增：引爆点 (x, z) 约束 ---
    x_coords = detonation_points[:, 0]
    z_coords = detonation_points[:, 2]
    if np.any((z_coords > 0.1 * x_coords + 80) | (z_coords < 0.1 * x_coords - 10)):
        return 1001.0
    
    # --- 新增：引爆点到视线距离约束 ---
    missile_pos_at_detonations = M1_INITIAL_POS + M1_VELOCITY * T_DETONATES[:, np.newaxis]
    target_center = np.array([0.0, 200.0, 5.0])
    
    vec_AP = detonation_points - missile_pos_at_detonations
    vec_AB = target_center - missile_pos_at_detonations
    
    norm_cross_product = np.linalg.norm(np.cross(vec_AP, vec_AB, axis=1), axis=1)
    norm_vec_AB = np.linalg.norm(vec_AB, axis=1)
    
    # 避免除以零
    norm_vec_AB[norm_vec_AB < 1e-9] = 1e-9
    
    distances = norm_cross_product / norm_vec_AB
    
    # --- 组合过滤条件 ---
    # 条件1: 引爆点到视线的距离必须小于等于60
    distance_mask = distances <= 60
    
    # # 条件2: 引爆点高度必须大于0
    # z_mask = detonation_points[:, 2] >= 0

    # # 合并两个掩码，得到最终有效的引爆点
    # valid_mask = np.logical_and(distance_mask, z_mask)
    valid_mask = distance_mask
    # 如果没有任何一个引爆点有效，则返回一个较小的负值
    if not np.any(valid_mask):
        return 101
        
    # 应用过滤，只保留有效的引爆点和对应的引爆时间
    detonation_points = detonation_points[valid_mask]
    T_DETONATES = T_DETONATES[valid_mask]
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
    
    # 设置仿真时间范围
    sim_start_time = np.min(T_DETONATES)
    sim_end_time = np.max(T_EFFECT_ENDS)
    time_steps = np.arange(sim_start_time, sim_end_time, SIM_DT)
    
    total_occlusion_time = 0.0
    
    for t in time_steps:
        missile_pos = get_missile_pos(t)
        if missile_pos[0] < 0:
            break
            
        active_mask = (t >= T_DETONATES) & (t < T_EFFECT_ENDS)
        if not np.any(active_mask):
            continue
            
        active_detonation_points = detonation_points[active_mask]
        active_T_DETONATES = T_DETONATES[active_mask]
        
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
    """为 pyswarms 准备的并行化目标函数"""
    with multiprocessing.Pool() as pool:
        results = pool.map(calculate_occlusion_time_single, swarm)
    return np.array(results)

# --- 新增：用于生成高质量初始粒子的辅助函数 ---
def check_drone_params_validity(params, drone_idx):
    """
    检查单个无人机的一组参数是否有效。
    有效性判断标准：引爆点到M1导弹视线的距离 <= 60。
    """
    a, v, t_take, t_fall = params

    drone_velocity = np.zeros(3)
    drone_velocity[0] = v * np.cos(a)
    drone_velocity[1] = v * np.sin(a)
    
    t_detonate = t_take + t_fall
    drone_pos_at_drop = DRONES_INITIAL_POS[drone_idx] + drone_velocity * t_take
    detonation_point = drone_pos_at_drop + drone_velocity * t_fall
    detonation_point[2] -= 0.5 * G * t_fall**2

    # 高度必须大于0
    if detonation_point[2] < 0:
        return False

    missile_pos_at_detonation = M1_INITIAL_POS + M1_VELOCITY * t_detonate
    target_center = np.array([0.0, 200.0, 5.0])
    
    vec_AP = detonation_point - missile_pos_at_detonation
    vec_AB = target_center - missile_pos_at_detonation
    
    norm_cross_product = np.linalg.norm(np.cross(vec_AP, vec_AB))
    norm_vec_AB = np.linalg.norm(vec_AB)
    
    if norm_vec_AB < 1e-9:
        norm_vec_AB = 1e-9
        
    distance = norm_cross_product / norm_vec_AB
    
    return distance <= 60

def generate_valid_solutions_for_drone(drone_idx, drone_bounds, n_solutions):
    """为单个无人机生成指定数量的有效解。"""
    low_b, high_b = drone_bounds
    valid_solutions = []
    while len(valid_solutions) < n_solutions:
        random_params = np.random.uniform(low=low_b, high=high_b, size=4)
        if check_drone_params_validity(random_params, drone_idx):
            valid_solutions.append(random_params)
    return np.array(valid_solutions)

# --- 4. 优化过程 ---

if __name__ == "__main__":
    # --- 为每架无人机定义不同的边界 [a, v, t_take, t_fall] ---
    # 您可以根据需要自定义以下边界值

    # FY1 的边界
    bounds_low_fy1 =  [np.pi - 0.2, 139, 0, 2]
    bounds_high_fy1 = [np.pi, 140, 4, 10]
    
    # FY2 的边界 (示例：速度较慢，投放时间窗口不同)
    bounds_low_fy2 =  [np.pi + 0.09, 70, 0, 6]
    bounds_high_fy2 = [np.pi + np.pi/2, 140, 26, 16.9]

    # FY3 的边界 (示例：速度较快，角度范围不同)
    bounds_low_fy3 =  [np.pi/2, 70, 0, 4]
    bounds_high_fy3 = [np.pi-0.48, 140, 40, 12]
    
    # 将各无人机的边界连接成一个12维的边界向量
    bounds_low = np.concatenate([bounds_low_fy1, bounds_low_fy2, bounds_low_fy3])
    bounds_high = np.concatenate([bounds_high_fy1, bounds_high_fy2, bounds_high_fy3])
    bounds = (bounds_low, bounds_high)

    # --- 新增：生成并注入高质量初始粒子 ---
    N_PARTICLES = 3000
    print("开始生成高质量初始粒子...")
    
    # 1. 为每架无人机生成50组满足距离约束的可行解
    n_valid_per_drone = 50
    print(f" - 正在为 FY1 生成 {n_valid_per_drone} 组可行解...")
    solutions_fy1 = generate_valid_solutions_for_drone(0, (bounds_low_fy1, bounds_high_fy1), n_valid_per_drone)
    print(f" - 正在为 FY2 生成 {n_valid_per_drone} 组可行解...")
    solutions_fy2 = generate_valid_solutions_for_drone(1, (bounds_low_fy2, bounds_high_fy2), n_valid_per_drone)
    print(f" - 正在为 FY3 生成 {n_valid_per_drone} 组可行解...")
    solutions_fy3 = generate_valid_solutions_for_drone(2, (bounds_low_fy3, bounds_high_fy3), n_valid_per_drone)

    # 2. 随机组合可行解以生成高质量粒子
    n_good_particles_target = 2000
    print(f" - 正在随机组合 {n_good_particles_target} 个高质量粒子...")
    good_particles_list = []
    for _ in range(n_good_particles_target):
        sol1 = solutions_fy1[np.random.randint(0, n_valid_per_drone)]
        sol2 = solutions_fy2[np.random.randint(0, n_valid_per_drone)]
        sol3 = solutions_fy3[np.random.randint(0, n_valid_per_drone)]
        particle = np.concatenate([sol1, sol2, sol3])
        good_particles_list.append(particle)
    good_particles = np.array(good_particles_list)

    # 3. 随机生成剩余的粒子
    n_random_particles = N_PARTICLES - n_good_particles_target
    print(f" - 已生成 {len(good_particles)} 个高质量粒子，正在随机生成剩余 {n_random_particles} 个粒子...")
    random_particles = np.random.uniform(low=bounds_low, high=bounds_high, size=(n_random_particles, 12))

    # 4. 合并成最终的初始粒子群
    init_pos = np.vstack([good_particles, random_particles])
    print(f"初始粒子群生成完毕，总数: {len(init_pos)}")
    
    print("="*60)
    print("开始为3架无人机各1枚烟幕弹进行多变量优化...")
    print("变量总数: 12")
    print("优化算法: 并行粒子群优化")
    print("计算将动用所有CPU核心，请稍候...")
    print("="*60)
    
    start_time = time.time()

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

    # 维度增加到12，并增加粒子数和迭代次数以应对更复杂的搜索空间
    optimizer = ps.single.GlobalBestPSO(n_particles=N_PARTICLES, dimensions=12, options=options, bounds=bounds, init_pos=init_pos)
    best_cost, best_params = optimizer.optimize(calculate_occlusion_time_parallel, iters=100, verbose=True)
    
    end_time = time.time()
    
    print("="*60)
    print("优化完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("-" * 60)
    
    max_occlusion_time = -best_cost
    
    print("找到的最佳结果:")
    params_per_drone = best_params.reshape(3, 4)
    
    for i in range(3):
        a, v, t_take, t_fall = params_per_drone[i]
        print(f"--- 无人机 FY{i+1} ---")
        print(f"  - 飞行方向角 (a): {np.rad2deg(a):.4f} 度")
        print(f"  - 飞行速度 (v): {v:.4f} m/s")
        print(f"  - 烟幕弹投放时间 (t_take): {t_take:.4f} s")
        print(f"  - 烟幕弹下落时间 (t_fall): {t_fall:.4f} s")
        if i < 2: print("-" * 30)

    print("\n" + "="*30)
    print(f"  => 最大遮蔽时间: {max_occlusion_time:.4f} 秒")
    print("=" * 30)
