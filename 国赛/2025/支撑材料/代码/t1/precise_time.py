import numpy as np

class KinematicObject:
    """运动学物体的基类"""
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)

class Missile(KinematicObject):
    """导弹类，匀速直线运动"""
    def __init__(self, initial_pos, target_pos, speed):
        super().__init__(initial_pos)
        self.target_pos = np.array(target_pos, dtype=float)
        self.speed = float(speed)
        
        direction = self.target_pos - self.pos
        self.velocity = direction / np.linalg.norm(direction) * self.speed

class Drone(KinematicObject):
    """无人机类，等高度匀速直线飞行"""
    def __init__(self, initial_pos, target_pos, speed):
        super().__init__(initial_pos)
        target_pos_2d = np.array([target_pos[0], target_pos[1]])
        pos_2d = np.array([self.pos[0], self.pos[1]])
        
        direction_2d = target_pos_2d - pos_2d
        unit_direction_2d = direction_2d / np.linalg.norm(direction_2d)
        velocity_2d = unit_direction_2d * speed
        self.velocity = np.array([velocity_2d[0], velocity_2d[1], 0.0])

TRUE_TARGET_POS = np.array([0, 200, 0])
FALSE_TARGET_POS = np.array([0, 0, 0])
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0
M1_INITIAL_POS = np.array([20000, 0, 2000], dtype=float)
M1_SPEED = 300
FY1_INITIAL_POS = np.array([17800, 0, 1800], dtype=float)
FY1_SPEED = 120
T_DROP = 1.5
T_DETONATE = T_DROP + 3.6
G = 9.80665
CLOUD_RADIUS = 10.0
SINK_VELOCITY = np.array([0.0, 0.0, -3.0])

missile_m1 = Missile(M1_INITIAL_POS, FALSE_TARGET_POS, M1_SPEED)
drone_fy1 = Drone(FY1_INITIAL_POS, FALSE_TARGET_POS, FY1_SPEED)
drone_pos_at_drop = drone_fy1.pos + drone_fy1.velocity * T_DROP
time_of_flight_to_detonation = T_DETONATE - T_DROP
detonation_point_offset = drone_fy1.velocity * time_of_flight_to_detonation
detonation_point_offset[2] -= 0.5 * G * time_of_flight_to_detonation**2
detonation_point = drone_pos_at_drop + detonation_point_offset

def get_missile_pos(t):
    return M1_INITIAL_POS + missile_m1.velocity * t

def get_cloud_pos(t):
    if t < T_DETONATE:
        return None
    time_since_detonation = t - T_DETONATE
    return detonation_point + SINK_VELOCITY * time_since_detonation

# --- 4. 遮挡判断逻辑 ---

def generate_cylinder_points(n_points=100):
    center_bottom = np.array([0, 200, 0])
    center_top = np.array([0, 200, 10])
    radius = 7.0
    angles = np.linspace(0, 2 * np.pi, n_points // 2, endpoint=False)
    bottom_points_x = center_bottom[0] + radius * np.cos(angles)
    bottom_points_y = center_bottom[1] + radius * np.sin(angles)
    bottom_points_z = np.full_like(bottom_points_x, center_bottom[2])
    top_points_x = center_top[0] + radius * np.cos(angles)
    top_points_y = center_top[1] + radius * np.sin(angles)
    top_points_z = np.full_like(top_points_x, center_top[2])
    all_points_x = np.concatenate([bottom_points_x, top_points_x])
    all_points_y = np.concatenate([bottom_points_y, top_points_y])
    all_points_z = np.concatenate([bottom_points_z, top_points_z])
    return np.vstack([all_points_x, all_points_y, all_points_z]).T

# 预先生成目标点以提高效率
TARGET_POINTS = generate_cylinder_points(1000)

def point_to_line_distance_sq(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return np.dot(point_vec, point_vec)
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1)
    closest_point = line_start + t * line_vec
    return np.sum((point - closest_point)**2)

def check_line_sphere_intersection(line_start, line_end, sphere_center, sphere_radius_sq):
    dist_sq = point_to_line_distance_sq(sphere_center, line_start, line_end)
    return dist_sq <= sphere_radius_sq

def check_occlusion_at_time(t):
    cloud_pos = get_cloud_pos(t)
    if cloud_pos is None:
        return False
    missile_pos = get_missile_pos(t)
    if np.sum((missile_pos - cloud_pos)**2) <= CLOUD_RADIUS**2:
        return True
    cloud_radius_sq = CLOUD_RADIUS**2
    for target_point in TARGET_POINTS:
        if not check_line_sphere_intersection(missile_pos, target_point, cloud_pos, cloud_radius_sq):
            return False
    return True

# --- 5. 二分查找函数 ---

def find_event_time(t_start, t_end, initial_state, precision=1e-5, max_iter=100):
    low = t_start
    high = t_end
    for _ in range(max_iter):
        if high - low < precision:
            break
        mid = (low + high) / 2
        if check_occlusion_at_time(mid) == initial_state:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# --- 6. 运行求解 ---

if __name__ == "__main__":
    print("开始使用二分法精确求解遮挡时间...")
    start_search_range = (7.5, 8.5)
    is_occluded_at_start_of_search = check_occlusion_at_time(start_search_range[0])
    if is_occluded_at_start_of_search:
        occlusion_start_time = "无法计算"
    else:
        occlusion_start_time = find_event_time(start_search_range[0], start_search_range[1], initial_state=False)
    
    end_search_range = (9.0, 10.0)
    is_occluded_at_end_of_search_start = check_occlusion_at_time(end_search_range[0])
    if not is_occluded_at_end_of_search_start:
         occlusion_end_time = "无法计算"
    else:
        occlusion_end_time = find_event_time(end_search_range[0], end_search_range[1], initial_state=True)

    print("计算结果:")
    if isinstance(occlusion_start_time, float):
         print(f"精确遮挡开始时间: {occlusion_start_time:.9f} s")
    if isinstance(occlusion_end_time, float):
        print(f"精确遮挡结束时间: {occlusion_end_time:.9f} s")
        duration = occlusion_end_time - occlusion_start_time
        print(f"遮挡持续时间: {duration:.9f} s")
