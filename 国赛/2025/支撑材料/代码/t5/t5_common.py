import itertools
import numpy as np
import multiprocessing

# 常量和参数定义
MISSILES_INITIAL_POS = np.array([
    [20000.0, 0.0, 2000.0],
    [19000.0, 600.0, 2100.0],
    [18000.0, -600.0, 1900.0]
])
MISSILES_TARGET_POS = np.array([0.0, 0.0, 0.0])
MISSILE_SPEED = 300.0

DRONES_INITIAL_POS = np.array([
    [17800.0, 0.0, 1800.0],
    [12000.0, 1400.0, 1400.0],
    [6000.0, -3000.0, 700.0],
    [11000.0, 2000.0, 1800.0],
    [13000.0, -2000.0, 1300.0]
])

CYLINDER_CENTER_BOTTOM = np.array([0.0, 200.0, 0.0])
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0

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
    t_numerator = np.sum(point_vec * line_vecs, axis=1)
    t = np.clip(t_numerator / line_lens_sq, 0, 1)
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
    missile_initial_pos = MISSILES_INITIAL_POS[missile_idx]
    missile_velocity = MISSILES_VELOCITY[missile_idx]
    missile_pos_at_detonations = missile_initial_pos + missile_velocity * t_detonates[:, np.newaxis]
    target_center = np.array([0.0, 200.0, 5.0])
    vec_AP = detonation_points - missile_pos_at_detonations
    vec_AB = target_center - missile_pos_at_detonations
    norm_cross_product = np.linalg.norm(np.cross(vec_AP, vec_AB, axis=1), axis=1)
    norm_vec_AB = np.linalg.norm(vec_AB, axis=1)
    norm_vec_AB[norm_vec_AB < 1e-9] = 1e-9
    distances = norm_cross_product / norm_vec_AB
    return distances

def calculate_union_of_intervals(intervals):
    if not intervals:
        return 0.0
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

def calculate_occlusion_time_single(params):
    params_per_drone = np.array(params).reshape(5, 8)
    
    for i in range(5):
        _a, _v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params_per_drone[i]
        if not (t_take1 + 2< t_take2 + 1< t_take3):
            return 200.0
        if (t_take1 + t_fall1 >= 60) or (t_take2 + t_fall2 >= 60) or (t_take3 + t_fall3 >= 60):
            return 300.0

    final_valid_detonation_points = []
    final_valid_t_detonates = []

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
        if np.any(combined_mask):
            final_valid_detonation_points.append(detonation_points[combined_mask])
            final_valid_t_detonates.append(T_DETONATES[combined_mask])

    if not final_valid_detonation_points:
        return 10.0
        
    detonation_points = np.vstack(final_valid_detonation_points)
    T_DETONATES = np.concatenate(final_valid_t_detonates)
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
    sim_start_time = np.min(T_DETONATES)
    sim_end_time = np.max(T_EFFECT_ENDS)
    time_steps = np.arange(sim_start_time, sim_end_time + SIM_DT, SIM_DT)
    
    if len(time_steps) == 0:
        return 0.0

    all_missiles_pos = np.array([get_missiles_pos(t) for t in time_steps])
    active_masks = (time_steps[:, np.newaxis] >= T_DETONATES) & (time_steps[:, np.newaxis] < T_EFFECT_ENDS)
    all_intervals = []

    for m_idx in range(3):
        occlusion_intervals = []
        is_currently_occluded = False
        occlusion_start_time = 0

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

def calculate_occlusion_time_parallel(swarm):
    with multiprocessing.Pool() as pool:
        results = pool.map(calculate_occlusion_time_single, swarm)
    return np.array(results)

def count_valid_grenades_t5(params, drone_idx):
    a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params

    if not (t_take1 + 2 < t_take2 + 1 < t_take3):
        return 0
    if (t_take1 + t_fall1 >= 60) or (t_take2 + t_fall2 >= 60) or (t_take3 + t_fall3 >= 60):
        return 0

    if drone_idx == 4 or drone_idx == 3:
        if not (t_fall1 < t_fall2 < t_fall3):
            return 0

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

    z_coords = detonation_points[:, 2]
    ground_mask = z_coords >= 0
    x_coords = detonation_points[:, 0]
    geo_mask = ~((z_coords > (21/190) * x_coords + 80) | (z_coords < 0.1 * x_coords - 10))
    
    if drone_idx == 0 or drone_idx == 4:
        distances = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
        distance_mask = distances <= 60
    else:
        dist1 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=0)
        dist2 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=1)
        dist3 = calculate_distances_to_los(detonation_points, T_DETONATES, missile_idx=2)
        min_distances = np.minimum(dist1, np.minimum(dist2, dist3))
        distance_mask = min_distances <= 60
    
    combined_mask = ground_mask & geo_mask & distance_mask
    return np.sum(combined_mask)

def generate_valid_solutions_for_drone_t5(drone_idx, drone_bounds, n_solutions):
    low_b, high_b = drone_bounds
    solutions_by_tier = {3: [], 2: [], 1: []}
    max_attempts = n_solutions * 10000
    
    for _ in range(max_attempts):
        if len(solutions_by_tier[3]) >= n_solutions:
            break
        random_params = np.random.uniform(low=low_b, high=high_b, size=8)
        valid_count = count_valid_grenades_t5(random_params, drone_idx)
        if valid_count in solutions_by_tier:
            solutions_by_tier[valid_count].append(random_params)

    final_solutions = []
    final_solutions.extend(solutions_by_tier[3])
    if len(final_solutions) < n_solutions:
        final_solutions.extend(solutions_by_tier[2])
    if len(final_solutions) < n_solutions:
        final_solutions.extend(solutions_by_tier[1])
    if len(final_solutions) == 0:
        return np.array([])
    final_solutions = final_solutions[:n_solutions]
    return np.array(final_solutions)

def generate_valid_solutions_for_drone_t5_normal(drone_idx, drone_bounds, n_solutions, mean_params, std_dev_divisor=10.0):
    low_b, high_b = drone_bounds
    std_dev = (np.array(high_b) - np.array(low_b)) / std_dev_divisor
    solutions_by_tier = {3: [], 2: [], 1: []}
    max_attempts = n_solutions * 4000

    for _ in range(max_attempts):
        if len(solutions_by_tier[3]) >= n_solutions:
            break
        random_params = np.random.normal(loc=mean_params, scale=std_dev, size=8)
        random_params = np.clip(random_params, low_b, high_b)
        valid_count = count_valid_grenades_t5(random_params, drone_idx)
        if valid_count in solutions_by_tier:
            solutions_by_tier[valid_count].append(random_params)

    final_solutions = []
    final_solutions.extend(solutions_by_tier[3])
    if len(final_solutions) < n_solutions:
        final_solutions.extend(solutions_by_tier[2])
    if len(final_solutions) < n_solutions:
        final_solutions.extend(solutions_by_tier[1])
    if len(final_solutions) == 0:
        return np.array([])
    final_solutions = final_solutions[:n_solutions]
    return np.array(final_solutions)

# 通用的边界定义
def get_drone_bounds():
    bounds_low_fy1 = [np.deg2rad(179.00), 139, 0, 0, 1, 0, 2, 0]
    bounds_high_fy1 = [np.deg2rad(179.96), 140, 14, 19.16, 14, 19.16, 14, 19.16]
    bounds_low_fy2 = [np.deg2rad(185.71), 70, 0, 0, 1, 0, 2, 0]
    bounds_high_fy2 = [np.deg2rad(353.49), 140, 51, 16.89, 51, 16.89, 51, 16.89]
    bounds_low_fy3 = [np.deg2rad(11.31), 70, 0, 0, 1, 0, 2, 0]
    bounds_high_fy3 = [np.deg2rad(151.93), 140, 60, 11.95, 60, 11.95, 60, 11.95]
    bounds_low_fy4 = [np.deg2rad(189.28), 70, 0, 0, 1, 0, 2, 0]
    bounds_high_fy4 = [np.deg2rad(350.06), 140, 57, 19.16, 57, 19.16, 57, 19.16]
    bounds_low_fy5 = [np.deg2rad(90.0), 70, 0, 0, 1, 0, 2, 0]
    bounds_high_fy5 = [np.deg2rad(170.41), 140, 45, 16.28, 45, 16.28, 45, 16.28]
    
    bounds_low = np.concatenate([bounds_low_fy1, bounds_low_fy2, bounds_low_fy3, bounds_low_fy4, bounds_low_fy5])
    bounds_high = np.concatenate([bounds_high_fy1, bounds_high_fy2, bounds_high_fy3, bounds_high_fy4, bounds_high_fy5])
    
    all_bounds = {
        0: (bounds_low_fy1, bounds_high_fy1), 1: (bounds_low_fy2, bounds_high_fy2),
        2: (bounds_low_fy3, bounds_high_fy3), 3: (bounds_low_fy4, bounds_high_fy4),
        4: (bounds_low_fy5, bounds_high_fy5),
    }
    
    return (bounds_low, bounds_high), all_bounds

# 参考最优解
OPTIMAL_PARAMS_REF = {
    0: np.array([np.deg2rad(179.5520), 139.2775, 1.0358, 4.2344, 4.6389, 5.7162, 13.0631, 8.1145]),
    1: np.array([np.deg2rad(289.7181), 92.8875, 9.0782, 2.1484, 12.3625, 8.8307, 26.0949, 2.8601]),
    2: np.array([np.deg2rad(81.2032), 92.1321, 21.6238, 2.6417, 30.6607, 2.9037, 51.3540, 7.1788]),
    3: np.array([np.deg2rad(283.4920), 114.3861, 4.1404, 10.3617, 9.4506, 10.6102, 27.7346, 10.5103]),
    4: np.array([np.deg2rad(110.7711), 116.0032, 15.0078, 3.5513, 21.0098, 11.9004, 33.9063, 5.8176]),
}
