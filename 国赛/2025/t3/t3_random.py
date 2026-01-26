import numpy as np
import time

# --- 1. 从 t3.py 复制的常量和核心计算函数 ---

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
SINK_VELOCITY = np.array([0.0, 0.0, -3.0])
SIM_DT = 0.01
CLOUD_DURATION = 20.0

# 预先计算导弹速度向量
m1_direction = M1_TARGET_POS - M1_INITIAL_POS
M1_VELOCITY = m1_direction / np.linalg.norm(m1_direction) * M1_SPEED

def get_missile_pos(t):
    return M1_INITIAL_POS + M1_VELOCITY * t

def generate_cylinder_points(n_points=100):
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
    line_vecs = lines_ends - lines_start
    point_vec = cloud_point - lines_start
    line_lens_sq = np.sum(line_vecs**2, axis=1)
    line_lens_sq[line_lens_sq == 0] = 1e-9
    t = np.clip(np.sum(point_vec * line_vecs, axis=1) / line_lens_sq, 0, 1)
    closest_points = lines_start + np.expand_dims(t, axis=1) * line_vecs
    return np.sum((closest_points - cloud_point)**2, axis=1)

def get_occluded_lines_mask_by_single_cloud(missile_pos, cloud_pos):
    if np.sum((missile_pos - cloud_pos)**2) <= CLOUD_RADIUS_SQ:
        return np.ones(len(TARGET_POINTS), dtype=bool)
    dist_sq_all_lines = vectorized_cloud_to_lines_distance_sq(cloud_pos, missile_pos, TARGET_POINTS)
    mask_width = dist_sq_all_lines <= CLOUD_RADIUS_SQ
    dist_sq_cloud_to_targets = np.sum((TARGET_POINTS - cloud_pos)**2, axis=1)
    dist_sq_missile_to_targets = np.sum((TARGET_POINTS - missile_pos)**2, axis=1)
    mask_between = dist_sq_cloud_to_targets - dist_sq_all_lines < dist_sq_missile_to_targets
    return np.logical_and(mask_width, mask_between)

# --- 2. 目标函数 (从 t3.py 复制) ---

def calculate_occlusion_time_single(params):
    a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params
    if t_take2 < t_take1 + 1 or t_take3 < t_take2 + 1:
        return 1000.0 
    if not (t_fall3 > t_fall2 > t_fall1):
        return 1000.0
    t_takes = np.array([t_take1, t_take2, t_take3])
    t_falls = np.array([t_fall1, t_fall2, t_fall3])
    drone_velocity = np.array([v * np.cos(a), v * np.sin(a), 0.0])
    T_DROPS = t_takes
    T_DETONATES = T_DROPS + t_falls
    drone_pos_at_drops = FY1_INITIAL_POS + drone_velocity * T_DROPS[:, np.newaxis]
    detonation_points = drone_pos_at_drops + drone_velocity * t_falls[:, np.newaxis]
    detonation_points[:, 2] -= 0.5 * G * t_falls**2
    valid_detonations = detonation_points[:, 2] >= 0
    if not np.any(valid_detonations):
        return 0
    detonation_points = detonation_points[valid_detonations]
    T_DETONATES = T_DETONATES[valid_detonations]
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
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
        occluded_lines_mask = np.zeros(len(TARGET_POINTS), dtype=bool)
        for cloud_pos in cloud_positions:
            current_cloud_mask = get_occluded_lines_mask_by_single_cloud(missile_pos, cloud_pos)
            occluded_lines_mask = np.logical_or(occluded_lines_mask, current_cloud_mask)
        is_occluded_at_t = np.all(occluded_lines_mask)
        if is_occluded_at_t:
             total_occlusion_time += SIM_DT
    return -total_occlusion_time

# --- 3. 随机搜索实现 ---

if __name__ == "__main__":
    # 定义8维参数的边界 (从 t3.py 复制)
    bounds_low = np.array([np.pi-0.0113, 70, 0, 0, 0, 0, 0, 0])
    bounds_high = np.array([np.pi, 140, 1, 5, 10, 10, 20, 20])

    # --- 以您提供的最优参数为中心进行正态分布随机取点 ---
    
    # 1. 设置最优参数为均值 (mean), 根据截图中的 best pos 设置
    best_params_mean = np.array([
        3.13545326, 139.832641, # a (rad), v
        0.001377,   3.617322,   # t_take1, t_fall1
        3.617426,   5.341025,   # t_take2, t_fall2
        5.553323,   6.028930    # t_take3, t_fall3
    ])

    # 2. 计算标准差 (区间长度的 1/30)
    std_devs = (bounds_high - bounds_low) / 30.0

    N_ITERATIONS = 100
    results = []
    
    print("="*60)
    print(f"开始以最优解为中心，进行 {N_ITERATIONS} 次正态分布随机搜索 (T3)...")
    print("="*60)
    start_time = time.time()

    for i in range(N_ITERATIONS):
        # 1. 在最优解附近根据正态分布生成一组新解
        random_params = np.random.normal(loc=best_params_mean, scale=std_devs, size=8)
        
        # 2. 将生成的值裁剪到合法边界内
        random_params = np.clip(random_params, bounds_low, bounds_high)
        
        # 3. 计算这组解的目标函数值
        cost = calculate_occlusion_time_single(random_params)
        
        # 目标函数返回的是负值，我们关心的是遮蔽时长本身
        occlusion_time = -cost
        
        # 过滤掉因不满足约束而返回的惩罚值
        if occlusion_time < -100:
             occlusion_time = 0.0

        results.append(occlusion_time)
        
        if (i + 1) % 10 == 0:
            print(f"  ...已完成 {i + 1}/{N_ITERATIONS} 次计算...")

    end_time = time.time()
    print("\n搜索完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("="*60)
    
    print("\n--- 随机搜索100次的结果 (协同遮蔽时间) ---")
    for i, res in enumerate(results):
        print(f" {res:.4f} 秒")
        
    print("\n" + "="*30)
    print("  统计信息:")
    print(f"  - 最大遮蔽时间: {np.max(results):.4f} 秒")
    print(f"  - 最小遮蔽时间: {np.min(results):.4f} 秒")
    print(f"  - 平均遮蔽时间: {np.mean(results):.4f} 秒")
    print("="*30)
