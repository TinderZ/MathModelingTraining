import itertools
import numpy as np
import time

# --- 1. 从 t5.py 复制的常量和参数定义 ---

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

# --- 2. 从 t5.py 复制的辅助和核心计算函数 ---

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

# --- 3. 主要功能函数 ---

def get_occlusion_intervals_with_details(params):
    """
    根据给定的40个参数，计算详细的遮蔽区间。

    Args:
        params (np.array or list): 长度为40的参数数组。

    Returns:
        list: 一个包含字典的列表，每个字典代表一个遮蔽区间，
              格式为 {'start': float, 'end': float, 'missile_idx': int, 'contributors': list[str]}
    """
    
    # 将40个参数分解到5架无人机，每架8个参数
    params_per_drone = np.array(params).reshape(5, 8)
    
    # --- 约束检查 ---
    for i in range(5):
        _a, _v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params_per_drone[i]
        if not (t_take1 + 2 < t_take2 + 1 < t_take3):
            print(f"警告: 无人机 {i+1} 的 t_take 约束不满足，可能无有效烟幕弹。")
        if (t_take1 + t_fall1 >= 60) or (t_take2 + t_fall2 >= 60) or (t_take3 + t_fall3 >= 60):
            print(f"警告: 无人机 {i+1} 的 t_take + t_fall 约束不满足，可能无有效烟幕弹。")

    # --- 计算并过滤每架无人机的有效烟幕弹 ---
    final_valid_grenades = []
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

        # 应用约束进行过滤
        z_coords = detonation_points[:, 2]
        ground_mask = z_coords >= 0
        x_coords = detonation_points[:, 0]
        geo_mask = ~((z_coords > (21/190) * x_coords + 80) | (z_coords < 0.1 * x_coords - 10))
        
        if i == 0 or i == 4:
            distances = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
            distance_mask = distances <= 60
        else:
            dist1 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
            dist2 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=1)
            dist3 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=2)
            min_distances = np.minimum(dist1, np.minimum(dist2, dist3))
            distance_mask = min_distances <= 60
        
        combined_mask = ground_mask & geo_mask & distance_mask

        # 将通过所有检查的烟幕弹及其来源信息加入最终列表
        valid_indices = np.where(combined_mask)[0]
        for grenade_idx in valid_indices:
            final_valid_grenades.append({
                'detonation_point': detonation_points[grenade_idx],
                't_detonate': T_DETONATES[grenade_idx],
                'drone_idx': i,
                'grenade_idx': grenade_idx 
            })

    # 如果没有任何有效的烟幕弹，则返回空列表
    if not final_valid_grenades:
        return []
        
    # 从列表中提取引爆点和引爆时间
    detonation_points = np.array([g['detonation_point'] for g in final_valid_grenades])
    T_DETONATES = np.array([g['t_detonate'] for g in final_valid_grenades])
    
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
    
    # 设置仿真时间范围
    sim_start_time = np.min(T_DETONATES)
    sim_end_time = np.max(T_EFFECT_ENDS)
    time_steps = np.arange(sim_start_time, sim_end_time + SIM_DT, SIM_DT)
    
    if len(time_steps) == 0:
        return []

    # --- 遮蔽区间计算 ---
    all_missiles_pos = np.array([get_missiles_pos(t) for t in time_steps])
    active_masks = (time_steps[:, np.newaxis] >= T_DETONATES) & (time_steps[:, np.newaxis] < T_EFFECT_ENDS)
    
    all_intervals_with_details = []

    # 为每枚导弹独立计算遮蔽区间
    for m_idx in range(3):
        is_currently_occluded = False
        occlusion_start_time = 0
        contributing_clouds_during_interval = set()

        for i, t in enumerate(time_steps):
            missile_pos = all_missiles_pos[i, m_idx]
            
            # 确定在当前时刻 t，哪些烟幕云是激活的
            active_mask_t = active_masks[i]
            if not np.any(active_mask_t):
                is_occluded_at_t = False
                contributing_grenade_indices = []
            else:
                active_indices_original = np.where(active_mask_t)[0]
                active_detonation_points_t = detonation_points[active_mask_t]
                active_T_DETONATES_t = T_DETONATES[active_mask_t]
                
                time_since_detonation = t - active_T_DETONATES_t
                cloud_positions = active_detonation_points_t + SINK_VELOCITY * time_since_detonation[:, np.newaxis]
                
                occluded_lines_mask = np.zeros(len(TARGET_POINTS), dtype=bool)
                contributing_grenade_indices = []
                
                for cloud_idx, cloud_pos in enumerate(cloud_positions):
                    current_cloud_mask = get_occluded_lines_mask_by_single_cloud(missile_pos, cloud_pos)
                    # 如果该云有贡献（遮挡了至少一条视线），则记录
                    if np.any(current_cloud_mask):
                        original_grenade_index = active_indices_original[cloud_idx]
                        contributing_grenade_indices.append(original_grenade_index)
                    occluded_lines_mask = np.logical_or(occluded_lines_mask, current_cloud_mask)
                
                is_occluded_at_t = np.all(occluded_lines_mask)

            # 状态变化检测
            if is_occluded_at_t:
                # 记录当前时刻有贡献的烟幕弹
                for grenade_idx in contributing_grenade_indices:
                    info = final_valid_grenades[grenade_idx]
                    contributing_clouds_during_interval.add(f"FY{info['drone_idx']+1}_G{info['grenade_idx']+1}")

            if is_occluded_at_t and not is_currently_occluded:
                is_currently_occluded = True
                occlusion_start_time = t
            elif not is_occluded_at_t and is_currently_occluded:
                is_currently_occluded = False
                all_intervals_with_details.append({
                    'start': occlusion_start_time,
                    'end': t,
                    'missile_idx': m_idx,
                    'contributors': sorted(list(contributing_clouds_during_interval))
                })
                contributing_clouds_during_interval = set()
        
        # 如果仿真结束时仍处于遮蔽状态，则记录最后一个区间
        if is_currently_occluded:
            all_intervals_with_details.append({
                'start': occlusion_start_time,
                'end': time_steps[-1],
                'missile_idx': m_idx,
                'contributors': sorted(list(contributing_clouds_during_interval))
            })
            
    return all_intervals_with_details


if __name__ == "__main__":
    # --- 使用您提供的最优参数集进行计算 ---
    best_params = [
        np.deg2rad(179.5824), 139.5909, 0.2246, 3.7469, 4.2000, 5.4670, 6.1853, 7.7078,  # FY1
        np.deg2rad(295.3677), 99.4268, 9.2210, 1.6224, 11.8339, 8.9548, 20.7638, 2.2823,  # FY2
        np.deg2rad(87.5943), 92.8446, 27.7529, 2.7758, 29.1748, 3.7963, 49.2416, 7.3458,  # FY3
        np.deg2rad(275.8299), 114.2396, 3.8243, 10.5066, 9.4303, 10.6933, 23.1460, 12.0854,  # FY4
        np.deg2rad(109.9545), 131.3048, 12.7298, 3.7228, 28.4380, 7.9242, 33.6368, 4.5380,  # FY5
    ]
    
    print("="*60)
    print("正在使用您提供的参数计算遮蔽区间...")
    print("="*60)
    
    start_time = time.time()
    detailed_intervals = get_occlusion_intervals_with_details(best_params)
    end_time = time.time()
    
    print(f"计算完成，耗时: {end_time - start_time:.4f} 秒\n")
    
    if not detailed_intervals:
        print("在此参数下，未找到任何有效的遮蔽区间。")
    else:
        print("找到的遮蔽区间详情如下:")
        # 按导弹编号和起始时间排序，方便查看
        sorted_intervals = sorted(detailed_intervals, key=lambda x: (x['missile_idx'], x['start']))
        
        current_missile = -1
        for interval in sorted_intervals:
            if interval['missile_idx'] != current_missile:
                current_missile = interval['missile_idx']
                print(f"\n--- 针对导弹 M{current_missile + 1} 的遮蔽区间 ---")
            
            duration = interval['end'] - interval['start']
            print(f"  - 时间: [{interval['start']:.2f}s, {interval['end']:.2f}s], "
                  f"时长: {duration:.2f}s")
            print(f"    贡献者: {', '.join(interval['contributors'])}")
            
    print("\n" + "="*60)
    print("脚本执行完毕。")
    print("="*60)
