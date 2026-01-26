import itertools
import numpy as np
import pyswarms as ps
import time
import multiprocessing

# --- 1. 常量和参数定义 ---

# 3枚来袭导弹 M1, M2, M3 的参数
MISSILES_INITIAL_POS = np.array([
    [20000.0, 0.0, 2000.0],    # M1
    [19000.0, 600.0, 2100.0],   # M2
    [18000.0, -600.0, 1900.0]  # M3
])
MISSILES_TARGET_POS = np.array([0.0, 0.0, 0.0]) # 目标统一为原点
MISSILE_SPEED = 300.0

# 5架无人机 FY1-FY5 的初始位置
DRONES_INITIAL_POS = np.array([
    [17800.0, 0.0, 1800.0],      # FY1
    [12000.0, 1400.0, 1400.0],   # FY2
    [6000.0, -3000.0, 700.0],    # FY3
    [11000.0, 2000.0, 1800.0],   # FY4
    [13000.0, -2000.0, 1300.0]   # FY5
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

# 预先计算3枚导弹的速度向量
missiles_directions = MISSILES_TARGET_POS - MISSILES_INITIAL_POS
missiles_norms = np.linalg.norm(missiles_directions, axis=1, keepdims=True)
MISSILES_VELOCITY = missiles_directions / missiles_norms * MISSILE_SPEED

def get_missiles_pos(t):
    """计算t时刻3枚导弹的位置"""
    return MISSILES_INITIAL_POS + MISSILES_VELOCITY * t

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
    if np.sum((missile_pos - cloud_pos)**2) <= CLOUD_RADIUS_SQ:
        return np.ones(len(TARGET_POINTS), dtype=bool)

    dist_sq_all_lines = vectorized_cloud_to_lines_distance_sq(cloud_pos, missile_pos, TARGET_POINTS)
    
    mask_width = dist_sq_all_lines <= CLOUD_RADIUS_SQ
    
    dist_sq_cloud_to_targets = np.sum((TARGET_POINTS - cloud_pos)**2, axis=1)
    dist_sq_missile_to_targets = np.sum((TARGET_POINTS - missile_pos)**2, axis=1)
    mask_between = dist_sq_cloud_to_targets - dist_sq_all_lines < dist_sq_missile_to_targets
    
    return np.logical_and(mask_width, mask_between)

def calculate_distances_to_los(detonation_points, t_detonates, missile_idx):
    """计算一组引爆点到指定导弹视线的距离"""
    missile_initial_pos = MISSILES_INITIAL_POS[missile_idx]
    missile_velocity = MISSILES_VELOCITY[missile_idx]
    missile_pos_at_detonations = missile_initial_pos + missile_velocity * t_detonates[:, np.newaxis]
    target_center = np.array([0.0, 200.0, 5.0]) # 目标中心点
    
    vec_AP = detonation_points - missile_pos_at_detonations
    vec_AB = target_center - missile_pos_at_detonations
    
    norm_cross_product = np.linalg.norm(np.cross(vec_AP, vec_AB, axis=1), axis=1)
    norm_vec_AB = np.linalg.norm(vec_AB, axis=1)
    norm_vec_AB[norm_vec_AB < 1e-9] = 1e-9
    
    distances = norm_cross_product / norm_vec_AB
    return distances

# --- 3. 目标函数 ---

def calculate_occlusion_time_single(params):
    """为单个粒子（一组40个参数）计算总遮蔽时间"""
    
    # 将40个参数分解到5架无人机，每架8个参数
    # [a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3]
    params_per_drone = np.array(params).reshape(5, 8)
    
    # --- 新增约束检查 ---
    for i in range(5):
        _a, _v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params_per_drone[i]
        # 约束1: t_take 必须严格递增
        if not (t_take1 + 2< t_take2 + 1< t_take3):
            return 200.0  # 返回一个较大的惩罚值
        # 约束2: t_take + t_fall < 60
        if (t_take1 + t_fall1 >= 60) or (t_take2 + t_fall2 >= 60) or (t_take3 + t_fall3 >= 60):
            return 300.0 # 返回一个较大的惩罚值

    final_valid_detonation_points = []
    final_valid_t_detonates = []

    # 分别计算并过滤每架无人机的烟幕弹
    for i in range(5):
        a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params_per_drone[i]
        
        t_takes = np.array([t_take1, t_take2, t_take3])
        t_falls = np.array([t_fall1, t_fall2, t_fall3])
        
        drone_velocity = np.array([v * np.cos(a), v * np.sin(a), 0.0])
        drone_initial_pos = DRONES_INITIAL_POS[i]

        T_DROPS = t_takes
        T_DETONATES = T_DROPS + t_falls
        
        drone_pos_at_drops = drone_initial_pos + drone_velocity * T_DROPS[:, np.newaxis]
        
        detonation_points = drone_pos_at_drops + drone_velocity * t_falls[:, np.newaxis]
        detonation_points[:, 2] -= 0.5 * G * t_falls**2
        
        if detonation_points.shape[0] == 0:
            continue

        # --- 应用约束进行过滤 ---
        
        # 1. 必须在地面以上引爆
        z_coords = detonation_points[:, 2]
        ground_mask = z_coords >= 0

        # 2. (x, z) 位置约束
        x_coords = detonation_points[:, 0]
        geo_mask = ~((z_coords > (21/190) * x_coords + 80) | (z_coords < 0.1 * x_coords - 10))
        
        # 3. 引爆点到导弹视线的距离约束
        if i == 0 or i == 4:  # FY1 (idx 0) 和 FY5 (idx 4)
            distances = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
            distance_mask = distances <= 60
        else:  # FY2 (idx 1), FY3 (idx 2), FY4 (idx 3)
            dist1 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
            dist2 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=1)
            dist3 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=2)
            min_distances = np.minimum(dist1, np.minimum(dist2, dist3))
            distance_mask = min_distances <= 60
        
        # 合并所有掩码
        combined_mask = ground_mask & geo_mask & distance_mask

        # 将通过所有检查的烟幕弹加入最终列表
        if np.any(combined_mask):
            final_valid_detonation_points.append(detonation_points[combined_mask])
            final_valid_t_detonates.append(T_DETONATES[combined_mask])

    # 如果没有任何有效的烟幕弹，则总遮蔽时间为0
    if not final_valid_detonation_points:
        return 10.0
        
    # 将所有有效的烟幕弹信息合并
    detonation_points = np.vstack(final_valid_detonation_points)
    T_DETONATES = np.concatenate(final_valid_t_detonates)
    
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
    
    # 设置仿真时间范围
    # 增加一个微小量以确保包含最后一个时间点
    sim_start_time = np.min(T_DETONATES)
    sim_end_time = np.max(T_EFFECT_ENDS)
    time_steps = np.arange(sim_start_time, sim_end_time + SIM_DT, SIM_DT)
    
    if len(time_steps) == 0:
        return 0.0

    # --- 新的遮蔽时间计算逻辑 ---
    
    # 预先计算所有时间步的导弹和烟幕云位置，以提高效率
    all_missiles_pos = np.array([get_missiles_pos(t) for t in time_steps])
    
    active_masks = (time_steps[:, np.newaxis] >= T_DETONATES) & (time_steps[:, np.newaxis] < T_EFFECT_ENDS)
    
    all_intervals = []

    # 1. 为每枚导弹独立计算遮蔽区间
    for m_idx in range(3): # 遍历三枚导弹
        occlusion_intervals = []
        is_currently_occluded = False
        occlusion_start_time = 0

        for i, t in enumerate(time_steps):
            missile_pos = all_missiles_pos[i, m_idx]

            # 确定在当前时刻 t，哪些烟幕云是激活的
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

            # 状态变化检测
            if is_occluded_at_t and not is_currently_occluded:
                is_currently_occluded = True
                occlusion_start_time = t
            elif not is_occluded_at_t and is_currently_occluded:
                is_currently_occluded = False
                occlusion_intervals.append((occlusion_start_time, t))
        
        # 如果仿真结束时仍处于遮蔽状态，则记录最后一个区间
        if is_currently_occluded:
            occlusion_intervals.append((occlusion_start_time, time_steps[-1]))
            
        all_intervals.extend(occlusion_intervals)

    # 2. 计算所有遮蔽区间的并集总时长
    total_occlusion_time = calculate_union_of_intervals(all_intervals)
             
    return -total_occlusion_time

def calculate_union_of_intervals(intervals):
    """计算一组时间区间的并集的总时长。"""
    if not intervals:
        return 0.0

    # 1. 按起始时间对区间进行排序
    intervals.sort(key=lambda x: x[0])

    merged = []
    if not intervals:
        return 0
        
    current_start, current_end = intervals[0]

    # 2. 遍历并合并重叠的区间
    for next_start, next_end in intervals[1:]:
        if next_start < current_end:  # 有重叠
            current_end = max(current_end, next_end)
        else:  # 无重叠，将前一个合并的区间存入列表
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    merged.append((current_start, current_end)) # 添加最后一个合并的区间

    # 3. 计算合并后区间的总时长
    total_duration = sum(end - start for start, end in merged)
    return total_duration

def calculate_occlusion_time_parallel(swarm):
    """为 pyswarms 准备的并行化目标函数"""
    with multiprocessing.Pool() as pool:
        results = pool.map(calculate_occlusion_time_single, swarm)
    return np.array(results)

# --- 新增：用于生成高质量初始粒子的辅助函数 ---
def count_valid_grenades_t5(params, drone_idx):
    """
    检查单个无人机的一组8维参数，并返回有效烟幕弹的数量。
    """
    a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params

    # 1. 检查内部约束
    if not (t_take1 + 2 < t_take2 + 1 < t_take3):
        return 0
    if (t_take1 + t_fall1 >= 60) or (t_take2 + t_fall2 >= 60) or (t_take3 + t_fall3 >= 60):
        return 0

    if drone_idx == 4 or drone_idx == 3:
        if not (t_fall1 < t_fall2 < t_fall3):
            return 0;

    # 2. 计算引爆点
    t_takes = np.array([t_take1, t_take2, t_take3])
    t_falls = np.array([t_fall1, t_fall2, t_fall3])
    
    drone_velocity = np.array([v * np.cos(a), v * np.sin(a), 0.0])
    drone_initial_pos = DRONES_INITIAL_POS[drone_idx]

    T_DROPS = t_takes
    T_DETONATES = T_DROPS + t_falls
    
    drone_pos_at_drops = drone_initial_pos + drone_velocity * T_DROPS[:, np.newaxis]
    
    detonation_points = drone_pos_at_drops + drone_velocity * t_falls[:, np.newaxis]
    detonation_points[:, 2] -= 0.5 * G * t_falls**2
    
    if detonation_points.shape[0] == 0:
        return 0

    # 3. 应用所有过滤条件
    z_coords = detonation_points[:, 2]
    ground_mask = z_coords >= 0

    x_coords = detonation_points[:, 0]
    geo_mask = ~((z_coords > (21/190) * x_coords + 80) | (z_coords < 0.1 * x_coords - 10))
    
    if drone_idx == 0 or drone_idx == 4:  # FY1, FY5
        distances = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
        distance_mask = distances <= 60
    else:  # FY2, FY3, FY4
        dist1 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
        dist2 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=1)
        dist3 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=2)
        min_distances = np.minimum(dist1, np.minimum(dist2, dist3))
        distance_mask = min_distances <= 60
    
    combined_mask = ground_mask & geo_mask & distance_mask

    # 4. 返回有效烟幕弹的数量
    return np.sum(combined_mask)

def generate_valid_solutions_for_drone_t5_normal(drone_idx, drone_bounds, n_solutions, mean_params, std_dev_divisor=10.0):
    """
    为单个无人机生成指定数量的有效8维解，使用正态分布进行抽样。
    """
    low_b, high_b = drone_bounds
    # 标准差是 n 分之一个区间长度
    std_dev = (np.array(high_b) - np.array(low_b)) / std_dev_divisor
    
    solutions_by_tier = {3: [], 2: [], 1: []}
    
    max_attempts = n_solutions * 3000  # 增加尝试次数以确保能找到足够的有效解
    print(f"    - 尝试最多 {max_attempts} 次来寻找解 (正态分布)...")

    for _ in range(max_attempts):
        if len(solutions_by_tier[3]) >= n_solutions:
            break
        
        # 从正态分布采样并裁剪到边界内
        random_params = np.random.normal(loc=mean_params, scale=std_dev, size=8)
        random_params = np.clip(random_params, low_b, high_b)
        
        valid_count = count_valid_grenades_t5(random_params, drone_idx)
        
        if valid_count in solutions_by_tier:
            solutions_by_tier[valid_count].append(random_params)

    # 按优先级组合最终的解列表
    final_solutions = []
    final_solutions.extend(solutions_by_tier[3])
    print(f"    - 找到 {len(solutions_by_tier[3])} 组 [3枚有效] 的解。")
    
    if len(final_solutions) < n_solutions:
        final_solutions.extend(solutions_by_tier[2])
        print(f"    - 找到 {len(solutions_by_tier[2])} 组 [2枚有效] 的解。当前总数: {len(final_solutions)}。")
    
    if len(final_solutions) < n_solutions:
        final_solutions.extend(solutions_by_tier[1])
        print(f"    - 找到 {len(solutions_by_tier[1])} 组 [1枚有效] 的解。当前总数: {len(final_solutions)}。")

    if len(final_solutions) == 0:
        return np.array([])

    # 截取所需数量的解
    final_solutions = final_solutions[:n_solutions]
    
    if len(final_solutions) < n_solutions:
        print(f"警告: 在所有层级搜索后，仅为无人机 FY{drone_idx+1} 找到 {len(final_solutions)}/{n_solutions} 组有效解。")

    return np.array(final_solutions)

def generate_valid_solutions_for_drone_t5(drone_idx, drone_bounds, n_solutions):
    """为单个无人机生成指定数量的有效8维解，优先选择有效烟幕弹数量更多的解。"""
    low_b, high_b = drone_bounds
    
    solutions_by_tier = {3: [], 2: [], 1: []}
    
    max_attempts = n_solutions * 10000  # 设置一个尝试上限
    print(f"    - 尝试最多 {max_attempts} 次来寻找解...")

    for _ in range(max_attempts):
        # 优化: 如果最高级别的解已经足够，则可以提前停止
        if len(solutions_by_tier[3]) >= n_solutions:
            break
        
        random_params = np.random.uniform(low=low_b, high=high_b, size=8)
        valid_count = count_valid_grenades_t5(random_params, drone_idx)
        
        if valid_count in solutions_by_tier:
            solutions_by_tier[valid_count].append(random_params)

    # 按优先级组合最终的解列表
    final_solutions = []
    final_solutions.extend(solutions_by_tier[3])
    print(f"    - 找到 {len(solutions_by_tier[3])} 组 [3枚有效] 的解。")
    
    if len(final_solutions) < n_solutions:
        final_solutions.extend(solutions_by_tier[2])
        print(f"    - 找到 {len(solutions_by_tier[2])} 组 [2枚有效] 的解。当前总数: {len(final_solutions)}。")
    
    if len(final_solutions) < n_solutions:
        final_solutions.extend(solutions_by_tier[1])
        print(f"    - 找到 {len(solutions_by_tier[1])} 组 [1枚有效] 的解。当前总数: {len(final_solutions)}。")

    if len(final_solutions) == 0:
        return np.array([]) # 返回空数组，让主程序处理错误

    # 截取所需数量的解
    final_solutions = final_solutions[:n_solutions]
    
    if len(final_solutions) < n_solutions:
        print(f"警告: 在所有层级搜索后，仅为无人机 FY{drone_idx+1} 找到 {len(final_solutions)}/{n_solutions} 组有效解。")

    return np.array(final_solutions)

# --- 4. 优化过程 ---

if __name__ == "__main__":
    # --- 为每架无人机定义不同的边界 [a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3] ---

    # FY1 
    bounds_low_fy1 =  [np.deg2rad(179.00), 139, 0, 0, 1, 0, 2, 0]
    bounds_high_fy1 = [np.deg2rad(179.96), 140, 14, 19.16, 14, 19.16, 14, 19.16]
    
    # FY2 
    bounds_low_fy2 =  [np.deg2rad(185.71), 70, 0, 0, 1, 0, 2, 0]
    bounds_high_fy2 = [np.deg2rad(353.49), 140, 51, 16.89, 51, 16.89, 51, 16.89]

    # FY3 
    bounds_low_fy3 =  [np.deg2rad(11.31), 70, 0, 0, 1, 0, 2, 0]
    bounds_high_fy3 = [np.deg2rad(151.93), 140, 60, 11.95, 60, 11.95, 60, 11.95]

    # FY4
    bounds_low_fy4 =  [np.deg2rad(189.28), 70, 0, 0, 1, 0, 2, 0]
    bounds_high_fy4 = [np.deg2rad(350.06), 140, 57, 19.16, 57, 19.16, 57, 19.16]

    # FY5 
    bounds_low_fy5 =  [np.deg2rad(90.0), 70, 0, 0, 1, 0, 2, 0]
    bounds_high_fy5 = [np.deg2rad(170.41), 140, 45, 16.28, 45, 16.28, 45, 16.28]
    
    # 将各无人机的边界连接成一个40维的边界向量
    bounds_low = np.concatenate([bounds_low_fy1, bounds_low_fy2, bounds_low_fy3, bounds_low_fy4, bounds_low_fy5])
    bounds_high = np.concatenate([bounds_high_fy1, bounds_high_fy2, bounds_high_fy3, bounds_high_fy4, bounds_high_fy5])
    bounds = (bounds_low, bounds_high)
    
    # --- 新增: 定义参考最优解 (来自初始运行) ---
    OPTIMAL_PARAMS_REF = {
        0: np.array([np.deg2rad(179.5520), 139.2775, 1.0358, 4.2344, 4.6389, 5.7162, 13.0631, 8.1145]),    # FY1
        1: np.array([np.deg2rad(289.7181), 92.8875, 9.0782, 2.1484, 12.3625, 8.8307, 26.0949, 2.8601]),   # FY2
        2: np.array([np.deg2rad(81.2032), 92.1321, 21.6238, 2.6417, 30.6607, 2.9037, 51.3540, 7.1788]),  # FY3
        3: np.array([np.deg2rad(283.4920), 114.3861, 4.1404, 10.3617, 9.4506, 10.6102, 27.7346, 10.5103]), # FY4
        4: np.array([np.deg2rad(110.7711), 116.0032, 15.0078, 3.5513, 21.0098, 11.9004, 33.9063, 5.8176]),# FY5
    }
    
    # --- 新增：分层优化高质量初始粒子生成 ---
    # 定义超参数
    STD_DEV_DIVISOR = 20.0 # 正态分布标准差的区间长度除数

    # 定义分层
    PRIMARY_DRONES = [3]  # 第1层: 优化 FY4
    SECONDARY_DRONES = [0, 1, 2, 4]

    # 定义粒子生成参数
    N_PRIMARY_SOLUTIONS = 100
    N_SECONDARY_SOLUTIONS_PER_DRONE = 50
    N_COMBINATIONS_PER_PRIMARY = 10
    N_PARTICLES = N_PRIMARY_SOLUTIONS * N_COMBINATIONS_PER_PRIMARY

    print("开始生成分层优化高质量初始粒子 (第1层: FY4)...")

    # 汇总所有无人机的边界，方便索引
    all_bounds = {
        0: (bounds_low_fy1, bounds_high_fy1), 1: (bounds_low_fy2, bounds_high_fy2),
        2: (bounds_low_fy3, bounds_high_fy3), 3: (bounds_low_fy4, bounds_high_fy4),
        4: (bounds_low_fy5, bounds_high_fy5),
    }

    drone_solutions = {}
    # 1. 为主要优化无人机生成解 (随机)
    for i in PRIMARY_DRONES:
        print(f" - 正在为主要目标 FY{i+1} 生成 {N_PRIMARY_SOLUTIONS} 组可行解 (随机)...")
        solutions = generate_valid_solutions_for_drone_t5(i, all_bounds[i], N_PRIMARY_SOLUTIONS)
        if solutions.shape[0] < N_PRIMARY_SOLUTIONS:
             print(f"警告: 主要目标无人机 FY{i+1} 只找到 {solutions.shape[0]}/{N_PRIMARY_SOLUTIONS} 组解。")
        if solutions.shape[0] == 0:
            raise ValueError(f"无法为主要目标无人机 FY{i+1} 找到任何可行解。")
        drone_solutions[i] = solutions

    # 2. 为次要优化无人机生成解 (正态分布, 带回退)
    for i in SECONDARY_DRONES:
        print(f" - 正在为次要目标 FY{i+1} 生成 {N_SECONDARY_SOLUTIONS_PER_DRONE} 组可行解 (正态分布)...")
        solutions = generate_valid_solutions_for_drone_t5_normal(i, all_bounds[i], N_SECONDARY_SOLUTIONS_PER_DRONE, OPTIMAL_PARAMS_REF[i], STD_DEV_DIVISOR)

        # 如果正态分布找不到解，回退到随机均匀分布
        if solutions.shape[0] == 0:
            print(f"警告: 正态分布未能为 FY{i+1} 找到任何解，回退到随机均匀分布...")
            solutions = generate_valid_solutions_for_drone_t5(i, all_bounds[i], N_SECONDARY_SOLUTIONS_PER_DRONE)

        if solutions.shape[0] < N_SECONDARY_SOLUTIONS_PER_DRONE:
             print(f"警告: 次要目标无人机 FY{i+1} 只找到 {solutions.shape[0]}/{N_SECONDARY_SOLUTIONS_PER_DRONE} 组解。")
        
        if solutions.shape[0] == 0:
            raise ValueError(f"无法为次要目标无人机 FY{i+1} 找到任何可行解（即使在回退后）。")
        drone_solutions[i] = solutions

    # 3. 组合解以生成高质量初始粒子
    print(f" - 正在组合 {N_PARTICLES} 个高质量粒子...")
    init_pos_list = []
    primary_solutions_fy4 = drone_solutions[3]

    for prim_sol in primary_solutions_fy4:
        for _ in range(N_COMBINATIONS_PER_PRIMARY):
            # 为每个次要无人机随机选择一个解
            sec_sol_fy1 = drone_solutions[0][np.random.randint(0, len(drone_solutions[0]))]
            sec_sol_fy2 = drone_solutions[1][np.random.randint(0, len(drone_solutions[1]))]
            sec_sol_fy3 = drone_solutions[2][np.random.randint(0, len(drone_solutions[2]))]
            sec_sol_fy5 = drone_solutions[4][np.random.randint(0, len(drone_solutions[4]))]
            
            # 按 FY1, FY2, FY3, FY4, FY5 的顺序拼接成一个40维粒子
            particle = np.concatenate([sec_sol_fy1, sec_sol_fy2, sec_sol_fy3, prim_sol, sec_sol_fy5])
            init_pos_list.append(particle)

    init_pos = np.array(init_pos_list)
    print(f"初始粒子群生成完毕，总数: {len(init_pos)}")
    
    print("="*60)
    print("开始为5架无人机对抗3枚导弹进行优化...")
    print("变量总数: 40")
    print("优化算法: 并行粒子群优化")
    print("计算将动用所有CPU核心，请耐心等待...")
    print("="*60)
    
    start_time = time.time()

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}

    # 维度增加到40，并增加粒子数以应对更复杂的搜索空间
    optimizer = ps.single.GlobalBestPSO(n_particles=N_PARTICLES, dimensions=40, options=options, bounds=bounds, init_pos=init_pos)
    best_cost, best_params = optimizer.optimize(calculate_occlusion_time_parallel, iters=100, verbose=True)
    
    end_time = time.time()
    
    print("="*60)
    print("优化完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("-" * 60)
    
    max_occlusion_time = -best_cost
    
    print("找到的最佳结果:")
    params_per_drone = best_params.reshape(5, 4, 2)
    
    for i in range(5):
        (a, v), (t_take1, t_fall1), (t_take2, t_fall2), (t_take3, t_fall3) = params_per_drone[i]
        print(f"--- 无人机 FY{i+1} ---")
        print(f"  - 飞行方向角 (a): {np.rad2deg(a):.4f} 度")
        print(f"  - 飞行速度 (v): {v:.4f} m/s")
        print(f"  - 烟幕弹 1: 投放时间 = {t_take1:.4f} s, 下落时间 = {t_fall1:.4f} s")
        print(f"  - 烟幕弹 2: 投放时间 = {t_take2:.4f} s, 下落时间 = {t_fall2:.4f} s")
        print(f"  - 烟幕弹 3: 投放时间 = {t_take3:.4f} s, 下落时间 = {t_fall3:.4f} s")
        if i < 4: print("-" * 30)

    print("\n" + "="*30)
    print(f"  => 最大协同遮蔽时间: {max_occlusion_time:.4f} 秒")
    print("=" * 30)