import numpy as np
import time
from t5_common import *

def get_occlusion_intervals_with_details(params):
    params_per_drone = np.array(params).reshape(5, 8)
    
    for i in range(5):
        _a, _v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params_per_drone[i]
        if not (t_take1 + 2 < t_take2 + 1 < t_take3):
            print(f"警告: 无人机 {i+1} 的 t_take 约束不满足，可能无有效烟幕弹。")
        if (t_take1 + t_fall1 >= 60) or (t_take2 + t_fall2 >= 60) or (t_take3 + t_fall3 >= 60):
            print(f"警告: 无人机 {i+1} 的 t_take + t_fall 约束不满足，可能无有效烟幕弹。")

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
        valid_indices = np.where(combined_mask)[0]
        for grenade_idx in valid_indices:
            final_valid_grenades.append({
                'detonation_point': detonation_points[grenade_idx],
                't_detonate': T_DETONATES[grenade_idx],
                'drone_idx': i,
                'grenade_idx': grenade_idx 
            })

    if not final_valid_grenades:
        return []
        
    detonation_points = np.array([g['detonation_point'] for g in final_valid_grenades])
    T_DETONATES = np.array([g['t_detonate'] for g in final_valid_grenades])
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
    sim_start_time = np.min(T_DETONATES)
    sim_end_time = np.max(T_EFFECT_ENDS)
    time_steps = np.arange(sim_start_time, sim_end_time + SIM_DT, SIM_DT)
    
    if len(time_steps) == 0:
        return []

    all_missiles_pos = np.array([get_missiles_pos(t) for t in time_steps])
    active_masks = (time_steps[:, np.newaxis] >= T_DETONATES) & (time_steps[:, np.newaxis] < T_EFFECT_ENDS)
    all_intervals_with_details = []

    for m_idx in range(3):
        is_currently_occluded = False
        occlusion_start_time = 0
        contributing_clouds_during_interval = set()

        for i, t in enumerate(time_steps):
            missile_pos = all_missiles_pos[i, m_idx]
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
                    if np.any(current_cloud_mask):
                        original_grenade_index = active_indices_original[cloud_idx]
                        contributing_grenade_indices.append(original_grenade_index)
                    occluded_lines_mask = np.logical_or(occluded_lines_mask, current_cloud_mask)
                
                is_occluded_at_t = np.all(occluded_lines_mask)

            if is_occluded_at_t:
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
        
        if is_currently_occluded:
            all_intervals_with_details.append({
                'start': occlusion_start_time,
                'end': time_steps[-1],
                'missile_idx': m_idx,
                'contributors': sorted(list(contributing_clouds_during_interval))
            })
            
    return all_intervals_with_details

if __name__ == "__main__":
    best_params = []
    
    print("正在使用参数计算遮蔽区间...")
    start_time = time.time()
    detailed_intervals = get_occlusion_intervals_with_details(best_params)
    end_time = time.time()
    
    print(f"计算完成，耗时: {end_time - start_time:.4f} 秒")
    
    if not detailed_intervals:
        print("在此参数下，未找到任何有效的遮蔽区间。")
    else:
        print("找到的遮蔽区间详情如下:")
        sorted_intervals = sorted(detailed_intervals, key=lambda x: (x['missile_idx'], x['start']))
        
        current_missile = -1
        for interval in sorted_intervals:
            if interval['missile_idx'] != current_missile:
                current_missile = interval['missile_idx']
                print(f"\n--- 针对导弹 M{current_missile + 1} 的遮蔽区间 ---")
            
            duration = interval['end'] - interval['start']
            print(f"时间: [{interval['start']:.2f}s, {interval['end']:.2f}s], 时长: {duration:.2f}s")
            print(f"贡献者: {', '.join(interval['contributors'])}")

    print("脚本执行完毕。")