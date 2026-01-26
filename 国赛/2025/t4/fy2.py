import numpy as np

# --- 1. 常量和参数定义 (与 t2.py 保持一致) ---

# 导弹M1参数
M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
M1_TARGET_POS = np.array([0.0, 0.0, 0.0])
M1_SPEED = 300.0

# 无人机FY1参数
FY1_INITIAL_POS = np.array([12000.0, 1400.0, 1400.0])

# 目标圆柱体参数
CYLINDER_CENTER_BOTTOM = np.array([0.0, 200.0, 0.0])
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0

# 物理和仿真参数
G = 9.80665
CLOUD_RADIUS = 10.0
SINK_VELOCITY = np.array([0.0, 0.0, -3.0])
SIM_DT = 0.01

# --- 2. 辅助和核心计算函数 (与 t2.py 保持一致) ---

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

def check_occlusion(missile_pos, cloud_pos):
    """检查在给定时刻是否完全遮挡"""
    if np.sum((missile_pos - cloud_pos)**2) <= CLOUD_RADIUS_SQ:
        return True

    dist_sq_all_lines = vectorized_cloud_to_lines_distance_sq(cloud_pos, missile_pos, TARGET_POINTS)
    is_occluded_by_width = np.all(dist_sq_all_lines <= CLOUD_RADIUS_SQ)
    if not is_occluded_by_width:
        return False

    dist_sq_cloud_to_targets = np.sum((TARGET_POINTS - cloud_pos)**2, axis=1)
    dist_sq_missile_to_targets = np.sum((TARGET_POINTS - missile_pos)**2, axis=1)
    is_between = np.all(dist_sq_cloud_to_targets - dist_sq_all_lines < dist_sq_missile_to_targets)
    
    return is_between

# --- 3. 计算遮挡时间的函数 ---

def calculate_and_print_occlusion_times(a_deg, v, t_take, t_fall):
    """
    根据输入的参数，计算并打印开始和结束遮挡的时间。
    """
    print("="*50)
    print("输入参数:")
    print(f"  - 飞行方向角 (a): {a_deg:.4f} 度")
    print(f"  - 飞行速度 (v): {v:.4f} m/s")
    print(f"  - 烟幕弹投放时间 (t_take): {t_take:.4f} s")
    print(f"  - 烟幕弹下落时间 (t_fall): {t_fall:.4f} s")
    print("-"*50)

    # 将角度转换为弧度
    a_rad = np.deg2rad(a_deg)
    
    # 计算引爆时间和烟幕弹失效时间
    T_DROP = t_take
    T_DETONATE = T_DROP + t_fall
    T_EFFECT_END = T_DETONATE + 20.0 # 烟幕持续20秒

    # 计算无人机和烟幕弹的运动轨迹
    drone_velocity = np.array([v * np.cos(a_rad), v * np.sin(a_rad), 0.0])
    drone_pos_at_drop = FY1_INITIAL_POS + drone_velocity * T_DROP
    detonation_point = drone_pos_at_drop + drone_velocity * t_fall
    detonation_point[2] -= 0.5 * G * t_fall**2
    
    # 检查引爆点是否在地面以上
    if detonation_point[2] < 0:
        print("计算结果: 烟幕弹在落地前未能引爆，无遮挡时间。")
        print("="*50)
        return

    # 初始化遮挡时间记录
    start_occlusion_time = None
    end_occlusion_time = None
    
    # 在烟幕弹生效的时间窗口内进行仿真
    time_steps = np.arange(T_DETONATE, T_EFFECT_END, SIM_DT)
    
    for t in time_steps:
        missile_pos = get_missile_pos(t)
        
        # 如果导弹已经飞过目标，则停止仿真
        if missile_pos[0] < 0:
            break

        # 计算当前时刻的烟幕云中心位置
        time_since_detonation = t - T_DETONATE
        cloud_pos = detonation_point + SINK_VELOCITY * time_since_detonation

        # 检查是否发生遮挡
        if check_occlusion(missile_pos, cloud_pos):
            # 如果是第一次检测到遮挡，记录开始时间
            if start_occlusion_time is None:
                start_occlusion_time = t
            # 持续更新最后遮挡时间
            end_occlusion_time = t
    
    print("计算结果:")
    if start_occlusion_time is not None:
        total_time = end_occlusion_time - start_occlusion_time + 0.01
        print(f"  - 开始遮挡时间: {start_occlusion_time:.4f} s")
        print(f"  - 结束遮挡时间: {end_occlusion_time:.4f} s")
        print(f"  - 总计遮挡时长: {total_time:.4f} s")
    else:
        print("  - 在此参数下，未检测到任何有效遮挡。")
    print("="*50)


if __name__ == "__main__":
    # --- 在这里输入您想测试的4个参数 ---
    # a_deg: 飞行方向角 (单位: 度)
    # v: 飞行速度 (单位: m/s)
    # t_take: 烟幕弹投放时间 (单位: s)
    # t_fall: 烟幕弹下落时间 (单位: s)
    
    # 示例参数 (您可以修改这些值)
    input_a_deg = 267.9904
    input_v = 108.5223
    input_t_take = 6.1151
    input_t_fall = 6.3542
    
    # 运行计算
    calculate_and_print_occlusion_times(
        a_deg=input_a_deg,
        v=input_v,
        t_take=input_t_take,
        t_fall=input_t_fall
    )
