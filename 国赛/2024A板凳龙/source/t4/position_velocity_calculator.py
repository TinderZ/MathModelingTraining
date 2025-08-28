import numpy as np
import pandas as pd
from pathlib import Path
from t4simulate import DragonRectangleSimulation

class PositionVelocityCalculator:
    """
    基于t4simulate的位置和速度计算器
    计算从-100s到100s每一秒的龙身坐标和速度
    """
    def __init__(self, simulation: DragonRectangleSimulation):
        self.sim = simulation
        self.head_speed = 100.0  # cm/s 龙头速度（恒定1m/s = 100cm/s）
        
    def get_tangent_direction(self, position, phase):
        """
        根据位置和相位计算轨迹切线方向（单位向量）
        position: 节点位置 [x, y]
        phase: 轨迹相位 'spiral_in', 'arc', 'spiral_out'
        """
        if phase == 'spiral_in':
            return self._get_spiral_in_tangent(position)
        elif phase == 'arc':
            return self._get_arc_tangent(position)
        elif phase == 'spiral_out':
            return self._get_spiral_out_tangent(position)
        else:
            return 1.0, 0.0  # 默认方向
    
    def _get_spiral_in_tangent(self, position):
        """计算盘入螺线的切线方向"""
        # 从位置反算theta
        r = np.linalg.norm(position)
        theta = r / self.sim.spiral_coeff
        
        a = self.sim.spiral_coeff
        # 计算切线向量分量
        dx_dtheta = a * (np.cos(theta) - theta * np.sin(theta))
        dy_dtheta = a * (np.sin(theta) + theta * np.cos(theta))
        
        # 归一化为单位向量
        magnitude = np.sqrt(dx_dtheta**2 + dy_dtheta**2)
        if magnitude > 1e-10:
            return dx_dtheta / magnitude, dy_dtheta / magnitude
        else:
            return 1.0, 0.0
    
    def _get_spiral_out_tangent(self, position):
        """计算盘出螺线的切线方向（与盘入螺线方向相反）"""
        # 盘出螺线是盘入螺线的中心对称，所以切线方向也要取反
        tangent_x, tangent_y = self._get_spiral_in_tangent(-position)  # 注意这里对position取反
        return -tangent_x, -tangent_y  # 切线方向也取反
    
    def _get_arc_tangent(self, position):
        """计算圆弧的切线方向"""
        # 判断该点在哪个圆弧上
        dist_to_O1 = np.linalg.norm(position - self.sim.O1)
        dist_to_O2 = np.linalg.norm(position - self.sim.O2)
        
        if abs(dist_to_O1 - self.sim.R1) < abs(dist_to_O2 - self.sim.R2):
            # 在O1圆弧上
            center = self.sim.O1
            radius = self.sim.R1
        else:
            # 在O2圆弧上
            center = self.sim.O2
            radius = self.sim.R2
        
        # 计算从圆心到点的向量
        radial_vector = position - center
        radial_magnitude = np.linalg.norm(radial_vector)
        
        if radial_magnitude > 1e-10:
            # 归一化径向向量
            radial_unit = radial_vector / radial_magnitude
            
            # 切线方向是径向向量逆时针旋转90度
            # (x, y) -> (-y, x)
            tangent_x = -radial_unit[1]
            tangent_y = radial_unit[0]
            
            return tangent_x, tangent_y
        else:
            return 1.0, 0.0
    
    def get_rod_direction(self, pos1, pos2):
        """
        计算连接两个节点的棍子方向（单位向量）
        从pos1指向pos2
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 1e-10:
            return dx / magnitude, dy / magnitude
        else:
            return 1.0, 0.0
    
    def calculate_angle_between_vectors(self, v1, v2):
        """
        计算两个向量之间的夹角（弧度）
        v1, v2: (x, y) 元组表示的向量
        """
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        # 限制dot_product在[-1, 1]范围内，避免数值误差
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return np.arccos(np.abs(dot_product))  # 取绝对值确保夹角在[0, π/2]
    
    def calculate_velocities_at_time(self):
        """
        计算当前时刻所有节点的速度大小
        返回: 速度数组，单位 cm/s
        """
        n_nodes = len(self.sim.positions)
        velocities = np.zeros(n_nodes)
        
        # 龙头速度已知
        velocities[0] = self.head_speed
        
        # 逐个计算后续节点的速度
        for i in range(n_nodes - 1):
            pos_current = self.sim.positions[i]
            pos_next = self.sim.positions[i + 1]
            phase_current = self.sim.node_phases[i]
            phase_next = self.sim.node_phases[i + 1]
            
            # 计算当前节点的切线方向（速度方向）
            tangent_current = self.get_tangent_direction(pos_current, phase_current)
            
            # 计算下一个节点的切线方向（速度方向）
            tangent_next = self.get_tangent_direction(pos_next, phase_next)
            
            # 计算棍子方向（从当前节点指向下一个节点）
            rod_direction = self.get_rod_direction(pos_current, pos_next)
            
            # 计算夹角
            angle1 = self.calculate_angle_between_vectors(tangent_current, rod_direction)
            angle2 = self.calculate_angle_between_vectors(tangent_next, rod_direction)
            
            # 应用动量守恒公式: v_current * cos(angle1) = v_next * cos(angle2)
            cos_angle1 = np.cos(angle1)
            cos_angle2 = np.cos(angle2)
            
            if cos_angle2 > 1e-10:  # 避免除零
                velocities[i + 1] = velocities[i] * cos_angle1 / cos_angle2
            else:
                # 如果cos_angle2接近0，说明速度方向与棍子垂直，速度为0
                velocities[i + 1] = 0.0
        
        return velocities

def compute_positions_and_velocities(start_time: int = -100, end_time: int = 100, 
                                   output_path: Path | None = None, pitch: float = 170.0):
    """
    计算从start_time到end_time每一秒的节点位置和速度，并写入Excel
    
    Args:
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）  
        output_path: 输出文件路径
        pitch: 螺距参数
    """
    print(f"开始计算从{start_time}s到{end_time}s的位置和速度数据...")
    
    # 创建仿真实例
    sim = DragonRectangleSimulation(pitch=pitch, debug=False)
    
    # 创建计算器
    calc = PositionVelocityCalculator(sim)
    
    # 时间步数
    time_steps = end_time - start_time + 1
    
    # 节点总数（包含龙头、龙身、龙尾、龙尾后）
    n_nodes = len(sim.positions)
    
    # 表头（列）为时间
    columns = [f"{t} s" for t in range(start_time, end_time + 1)]
    
    # 位置数据矩阵：行数 = 2*n_nodes（x和y坐标），列数 = time_steps
    position_data = np.zeros((2 * n_nodes, time_steps), dtype=float)
    
    # 速度数据矩阵：行数 = n_nodes，列数 = time_steps  
    velocity_data = np.zeros((n_nodes, time_steps), dtype=float)
    
    # 行索引标签
    position_labels = []
    velocity_labels = []
    
    # 构建标签 - 224个节点：龙头(0) + 第1-221节龙身(1-221) + 龙尾(222) + 龙尾后(223)
    position_labels.extend(["龙头x (m)", "龙头y (m)"])
    velocity_labels.append("龙头 (m/s)")
    
    # 第1到第221节龙身
    for i in range(1, 222):
        position_labels.extend([f"第{i}节龙身x (m)", f"第{i}节龙身y (m)"])
        velocity_labels.append(f"第{i}节龙身 (m/s)")
    
    # 龙尾节点 (第222个节点)
    position_labels.extend(["龙尾x (m)", "龙尾y (m)"])
    velocity_labels.append("龙尾 (m/s)")
    
    # 龙尾（后）节点 (第223个节点)
    position_labels.extend(["龙尾（后）x (m)", "龙尾（后）y (m)"])
    velocity_labels.append("龙尾（后） (m/s)")
    
    # 设置仿真的初始时间
    sim.time = start_time
    sim.update_all_positions()
    
    # 按时间步计算
    for t_idx, t in enumerate(range(start_time, end_time + 1)):
        # 更新仿真到当前时间
        sim.time = t
        sim.update_all_positions()
        
        # 计算位置（转换为米）
        for node_idx in range(n_nodes):
            position_data[2 * node_idx, t_idx] = sim.positions[node_idx][0] / 100.0  # x坐标，cm转m
            position_data[2 * node_idx + 1, t_idx] = sim.positions[node_idx][1] / 100.0  # y坐标，cm转m
        
        # 计算速度（转换为m/s）
        velocities_cm_s = calc.calculate_velocities_at_time()
        velocity_data[:, t_idx] = velocities_cm_s / 100.0  # cm/s转m/s
        
        # 进度显示
        if (t_idx + 1) % 20 == 0 or t_idx == time_steps - 1:
            print(f"已完成 {t_idx + 1}/{time_steps} 个时间步的计算")
    
    # 默认输出到当前目录
    if output_path is None:
        output_path = Path("dragon_trajectory_results.xlsx")
    
    # 保留6位小数
    position_data = np.round(position_data, 6)
    velocity_data = np.round(velocity_data, 6)
    
    # 创建DataFrame
    position_df = pd.DataFrame(data=position_data, index=position_labels, columns=columns)
    velocity_df = pd.DataFrame(data=velocity_data, index=velocity_labels, columns=columns)
    
    # 写入Excel
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            position_df.to_excel(writer, sheet_name="位置")
            velocity_df.to_excel(writer, sheet_name="速度")
        print(f"计算结果已保存到：{output_path}")
    except Exception as e:
        print(f"保存Excel文件时出错：{e}")
        # 尝试保存为CSV作为备用
        position_df.to_csv(output_path.with_suffix('.positions.csv'))
        velocity_df.to_csv(output_path.with_suffix('.velocities.csv'))
        print(f"已保存为CSV文件：{output_path.with_suffix('.positions.csv')} 和 {output_path.with_suffix('.velocities.csv')}")
    
    return position_df, velocity_df

if __name__ == "__main__":
    # 运行计算：从-100s到100s
    compute_positions_and_velocities(start_time=-100, end_time=100, pitch=170.0)
