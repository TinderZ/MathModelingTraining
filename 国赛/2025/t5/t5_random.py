import numpy as np
import time

# --- 1. 从 t5.py 复制的常量和核心计算函数 ---

# 导弹参数
MISSILES_INITIAL_POS = np.array([
    [20000.0, 0.0, 2000.0], [19000.0, 600.0, 2100.0], [18000.0, -600.0, 1900.0]
])
MISSILES_TARGET_POS = np.array([0.0, 0.0, 0.0])
MISSILE_SPEED = 300.0

# 无人机初始位置
DRONES_INITIAL_POS = np.array([
    [17800.0, 0.0, 1800.0], [12000.0, 1400.0, 1400.0], [6000.0, -3000.0, 700.0],
    [11000.0, 2000.0, 1800.0], [13000.0, -2000.0, 1300.0]
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

# 预计算
missiles_directions = MISSILES_TARGET_POS - MISSILES_INITIAL_POS
missiles_norms = np.linalg.norm(missiles_directions, axis=1, keepdims=True)
MISSILES_VELOCITY = missiles_directions / missiles_norms * MISSILE_SPEED
CLOUD_RADIUS_SQ = CLOUD_RADIUS**2

def get_missiles_pos(t):
    return MISSILES_INITIAL_POS + MISSILES_VELOCITY * t

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

def calculate_distances_to_los(detonation_points, t_detonates, missile_idx):
    missile_pos_at_detonations = MISSILES_INITIAL_POS[missile_idx] + MISSILES_VELOCITY[missile_idx] * t_detonates[:, np.newaxis]
    target_center = np.array([0.0, 200.0, 5.0])
    vec_AP = detonation_points - missile_pos_at_detonations
    vec_AB = target_center - missile_pos_at_detonations
    norm_cross_product = np.linalg.norm(np.cross(vec_AP, vec_AB, axis=1), axis=1)
    norm_vec_AB = np.linalg.norm(vec_AB, axis=1)
    norm_vec_AB[norm_vec_AB < 1e-9] = 1e-9
    return norm_cross_product / norm_vec_AB

def calculate_union_of_intervals(intervals):
    if not intervals: return 0.0
    intervals.sort(key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start < current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    return sum(end - start for start, end in merged)

# --- 2. 目标函数 (从 t5.py 复制并简化) ---

def calculate_occlusion_time_single(params):
    params_per_drone = np.array(params).reshape(5, 8)
    
    # 内部约束检查
    for i in range(5):
        _a, _v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params_per_drone[i]
        if not (t_take1 + 2 < t_take2 + 1 < t_take3): return 200.0
        if (t_take1 + t_fall1 >= 60) or (t_take2 + t_fall2 >= 60) or (t_take3 + t_fall3 >= 60): return 300.0

    final_valid_detonation_points = []
    final_valid_t_detonates = []

    # 计算有效烟幕弹
    for i in range(5):
        a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params_per_drone[i]
        t_takes = np.array([t_take1, t_take2, t_take3])
        t_falls = np.array([t_fall1, t_fall2, t_fall3])
        drone_velocity = np.array([v * np.cos(a), v * np.sin(a), 0.0])
        T_DROPS = t_takes
        T_DETONATES = T_DROPS + t_falls
        drone_pos_at_drops = DRONES_INITIAL_POS[i] + drone_velocity * T_DROPS[:, np.newaxis]
        detonation_points = drone_pos_at_drops + drone_velocity * t_falls[:, np.newaxis]
        detonation_points[:, 2] -= 0.5 * G * t_falls**2
        
        if detonation_points.shape[0] == 0: continue
        
        ground_mask = detonation_points[:, 2] >= 0
        x_coords = detonation_points[:, 0]
        geo_mask = ~((detonation_points[:, 2] > (21/190) * x_coords + 80) | (detonation_points[:, 2] < 0.1 * x_coords - 10))
        
        if i == 0 or i == 4:
            distance_mask = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0) <= 60
        else:
            dist1 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
            dist2 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=1)
            dist3 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=2)
            distance_mask = np.minimum(dist1, np.minimum(dist2, dist3)) <= 60
            
        combined_mask = ground_mask & geo_mask & distance_mask
        if np.any(combined_mask):
            final_valid_detonation_points.append(detonation_points[combined_mask])
            final_valid_t_detonates.append(T_DETONATES[combined_mask])

    if not final_valid_detonation_points: return 10.0
        
    detonation_points = np.vstack(final_valid_detonation_points)
    T_DETONATES = np.concatenate(final_valid_t_detonates)
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
    
    sim_start_time, sim_end_time = np.min(T_DETONATES), np.max(T_EFFECT_ENDS)
    time_steps = np.arange(sim_start_time, sim_end_time + SIM_DT, SIM_DT)
    
    if len(time_steps) == 0: return 0.0

    all_missiles_pos = np.array([get_missiles_pos(t) for t in time_steps])
    active_masks = (time_steps[:, np.newaxis] >= T_DETONATES) & (time_steps[:, np.newaxis] < T_EFFECT_ENDS)
    
    all_intervals = []
    for m_idx in range(3):
        occlusion_intervals, is_currently_occluded, occlusion_start_time = [], False, 0
        for i, t in enumerate(time_steps):
            missile_pos = all_missiles_pos[i, m_idx]
            active_mask_t = active_masks[i]
            if not np.any(active_mask_t):
                is_occluded_at_t = False
            else:
                active_detonation_points_t = detonation_points[active_mask_t]
                active_T_DETONATES_t = T_DETONATES[active_mask_t]
                time_since_detonation = t - active_T_DETONATES_t
                cloud_positions = active_detonation_points_t + SINK_VELOCITY * time_since_detonation[:, np.newaxis]
                occluded_lines_mask = np.zeros(len(TARGET_POINTS), dtype=bool)
                for cloud_pos in cloud_positions:
                    current_cloud_mask = get_occluded_lines_mask_by_single_cloud(missile_pos, cloud_pos)
                    occluded_lines_mask = np.logical_or(occluded_lines_mask, current_cloud_mask)
                is_occluded_at_t = np.all(occluded_lines_mask)
            
            if is_occluded_at_t and not is_currently_occluded:
                is_currently_occluded = True
                occlusion_start_time = t
            elif not is_occluded_at_t and is_currently_occluded:
                is_currently_occluded = False
                occlusion_intervals.append((occlusion_start_time, t))
        
        if is_currently_occluded:
            occlusion_intervals.append((occlusion_start_time, time_steps[-1]))
        all_intervals.extend(occlusion_intervals)

    total_occlusion_time = calculate_union_of_intervals(all_intervals)
    return -total_occlusion_time


# --- 3. 随机搜索实现 ---

if __name__ == "__main__":
    # 定义40维参数的边界 (从 t5.py 复制)
    bounds_low = np.concatenate([
        [np.deg2rad(179.00), 139, 0, 0, 1, 0, 2, 0],       # FY1
        [np.deg2rad(185.71), 70, 0, 0, 1, 0, 2, 0],        # FY2
        [np.deg2rad(11.31), 70, 0, 0, 1, 0, 2, 0],         # FY3
        [np.deg2rad(189.28), 70, 0, 0, 1, 0, 2, 0],        # FY4
        [np.deg2rad(90.0), 70, 0, 0, 1, 0, 2, 0]           # FY5
    ])
    bounds_high = np.concatenate([
        [np.deg2rad(179.96), 140, 14, 19.16, 14, 19.16, 14, 19.16], # FY1
        [np.deg2rad(353.49), 140, 51, 16.89, 51, 16.89, 51, 16.89], # FY2
        [np.deg2rad(151.93), 140, 60, 11.95, 60, 11.95, 60, 11.95], # FY3
        [np.deg2rad(350.06), 140, 57, 19.16, 57, 19.16, 57, 19.16], # FY4
        [np.deg2rad(170.41), 140, 45, 16.28, 45, 16.28, 45, 16.28]  # FY5
    ])

    # --- 新增：以您提供的最优参数为中心进行正态分布随机取点 ---
    
    # 1. 设置最优参数为均值 (mean)
    best_params_mean = np.array([
        np.deg2rad(179.5824), 139.5909, 0.2246, 3.7469, 4.2000, 5.4670, 6.1853, 7.7078,    # FY1
        np.deg2rad(295.3677), 99.4268, 9.2210, 1.6224, 11.8339, 8.9548, 20.7638, 2.2823,   # FY2
        np.deg2rad(87.5943), 92.8446, 27.7529, 2.7758, 29.1748, 3.7963, 49.2416, 7.3458,   # FY3
        np.deg2rad(275.8299), 114.2396, 3.8243, 10.5066, 9.4303, 10.6933, 23.1460, 12.0854,  # FY4
        np.deg2rad(109.9545), 131.3048, 12.7298, 3.7228, 28.4380, 7.9242, 33.6368, 4.5380,  # FY5
    ])

    # 2. 计算标准差 (区间长度的 1/20)
    std_devs = (bounds_high - bounds_low) / 30.0

    N_ITERATIONS = 100
    results = []
    
    print("="*60)
    print(f"开始以最优解为中心，进行 {N_ITERATIONS} 次正态分布随机搜索...")
    print("="*60)
    start_time = time.time()

    for i in range(N_ITERATIONS):
        # 1. 在最优解附近根据正态分布生成一组新解
        random_params = np.random.normal(loc=best_params_mean, scale=std_devs, size=40)
        
        # 2. 将生成的值裁剪到合法边界内
        random_params = np.clip(random_params, bounds_low, bounds_high)
        
        # 3. 计算这组解的目标函数值
        cost = calculate_occlusion_time_single(random_params)
        
        # 目标函数返回的是负值，我们关心的是遮蔽时长本身
        occlusion_time = -cost
        
        # 过滤掉因不满足约束而返回的惩罚值
        if occlusion_time < -100:
             # 如果需要，可以打印出来观察
             # print(f"第 {i+1:3d} 次: 无效解 (惩罚值: {-occlusion_time:.1f})")
             # 为简单起见，我们将无效解的遮蔽时间记为0
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
