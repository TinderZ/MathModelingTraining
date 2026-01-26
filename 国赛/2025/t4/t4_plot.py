import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. 常量和参数定义 (从 t4/t4.py 复制) ---

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# --- 2. 辅助和核心计算函数 (从 t4/t4.py 复制) ---

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

# --- 3. 可视化核心函数 ---

def get_intervals(time_steps, status_history):
    """将布尔状态历史记录转换为 (起始时间, 持续时间) 的区间列表"""
    intervals = []
    in_interval = False
    start_time = 0
    for i, status in enumerate(status_history):
        if status and not in_interval:
            in_interval = True
            start_time = time_steps[i]
        elif not status and in_interval:
            in_interval = False
            duration = time_steps[i-1] - start_time + SIM_DT
            intervals.append((start_time, duration))
    if in_interval: # 处理最后一个区间
        duration = time_steps[-1] - start_time + SIM_DT
        intervals.append((start_time, duration))
    return intervals

def visualize_occlusion_timeline(params):
    """
    根据给定的12个参数，模拟遮挡过程并生成时间轴可视化图。
    """
    params_per_drone = np.array(params).reshape(3, 4)
    drone_velocities = np.zeros((3, 3))
    drone_velocities[:, 0] = params_per_drone[:, 1] * np.cos(params_per_drone[:, 0])
    drone_velocities[:, 1] = params_per_drone[:, 1] * np.sin(params_per_drone[:, 0])
    
    t_takes = params_per_drone[:, 2]
    t_falls = params_per_drone[:, 3]

    T_DROPS = t_takes
    T_DETONATES = T_DROPS + t_falls
    
    drone_pos_at_drops = DRONES_INITIAL_POS + drone_velocities * T_DROPS[:, np.newaxis]
    detonation_points = drone_pos_at_drops + drone_velocities * t_falls[:, np.newaxis]
    detonation_points[:, 2] -= 0.5 * G * t_falls**2
    
    valid_mask = detonation_points[:, 2] >= 0
    if not np.any(valid_mask):
        print("所有烟幕弹都在地面以下引爆，无遮挡。")
        return

    detonation_points = detonation_points[valid_mask]
    T_DETONATES = T_DETONATES[valid_mask]
    T_EFFECT_ENDS = T_DETONATES + CLOUD_DURATION
    
    sim_start_time = np.min(T_DETONATES)
    sim_end_time = np.max(T_EFFECT_ENDS)
    time_steps = np.arange(sim_start_time, sim_end_time, SIM_DT)
    
    # --- 数据记录 ---
    total_occlusion_history = []
    drone_contribution_history = {i: [] for i in range(3)}

    for t in time_steps:
        missile_pos = get_missile_pos(t)
        if missile_pos[0] < 0:
            break
            
        active_mask = (t >= T_DETONATES) & (t < T_EFFECT_ENDS)
        if not np.any(active_mask):
            total_occlusion_history.append(False)
            for i in range(3): drone_contribution_history[i].append(False)
            continue
            
        active_indices = np.where(active_mask)[0]
        active_detonation_points = detonation_points[active_mask]
        active_T_DETONATES = T_DETONATES[active_mask]
        
        time_since_detonation = t - active_T_DETONATES
        cloud_positions = active_detonation_points + SINK_VELOCITY * time_since_detonation[:, np.newaxis]
        
        occluded_lines_mask = np.zeros(len(TARGET_POINTS), dtype=bool)
        current_contributions = {i: False for i in range(3)}

        for i, cloud_pos in zip(active_indices, cloud_positions):
            current_cloud_mask = get_occluded_lines_mask_by_single_cloud(missile_pos, cloud_pos)
            if np.any(current_cloud_mask):
                current_contributions[i] = True
            occluded_lines_mask = np.logical_or(occluded_lines_mask, current_cloud_mask)
        
        is_occluded_at_t = np.all(occluded_lines_mask)
        total_occlusion_history.append(is_occluded_at_t)
        for i in range(3):
            drone_contribution_history[i].append(current_contributions.get(i, False))

    # --- 绘图 ---
    fig, axs = plt.subplots(4, 1, figsize=(15, 6), sharex=True)
    
    y_labels = ["总遮挡", "FY3 贡献", "FY2 贡献", "FY1 贡献"]
    colors = ['tab:red', 'tab:purple', 'tab:blue', 'tab:green']

    # 绘制总遮挡时间轴
    total_intervals = get_intervals(time_steps, total_occlusion_history)
    axs[0].broken_barh(total_intervals, (0.4, 0.2), facecolors=colors[0])
    
    # 绘制各无人机贡献时间轴
    for i in range(3):
        drone_intervals = get_intervals(time_steps, drone_contribution_history[2-i])
        axs[i+1].broken_barh(drone_intervals, (0.4, 0.2), facecolors=colors[i+1])

    for i, ax in enumerate(axs):
        ax.set_yticks([0.5])
        ax.set_yticklabels([y_labels[i]])
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    axs[-1].set_xlabel("仿真时间 (秒)")
    fig.suptitle("无人机协同遮挡时间轴分析", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    # --- 在这里输入您从 t4.py 优化得到的12个最佳参数 ---
    # 这是一组示例参数，请替换为您自己的结果
    best_params = [
        # FY1: a, v, t_take, t_fall
        2.99, 135.0, 1.5, 3.0,
        # FY2: a, v, t_take, t_fall
        3.5, 120.0, 10.0, 8.0,
        # FY3: a, v, t_take, t_fall
        2.0, 100.0, 30.0, 10.0
    ]
    
    print("正在生成可视化图表...")
    visualize_occlusion_timeline(best_params)
    print("图表已生成。")
