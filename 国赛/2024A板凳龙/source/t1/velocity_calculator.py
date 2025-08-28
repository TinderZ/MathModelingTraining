import numpy as np
import pandas as pd
from pathlib import Path
from t1simulate import DragonSimulation

class VelocityCalculator:
    def __init__(self, simulation: DragonSimulation):
        self.sim = simulation
        self.head_speed = 100.0  # cm/s 龙头速度
        
    def get_tangent_direction(self, theta):
        """
        计算阿基米德螺旋线在角度theta处的切线方向（单位向量）
        阿基米德螺旋线: r = a * theta
        参数方程: x = a*theta*cos(theta), y = a*theta*sin(theta)
        切线向量: dx/dtheta, dy/dtheta
        """
        a = self.sim.spiral_coeff
        
        # 计算切线向量分量
        dx_dtheta = a * (np.cos(theta) - theta * np.sin(theta))
        dy_dtheta = a * (np.sin(theta) + theta * np.cos(theta))
        
        # 归一化为单位向量
        magnitude = np.sqrt(dx_dtheta**2 + dy_dtheta**2)
        if magnitude > 1e-10:  # 避免除零
            return dx_dtheta / magnitude, dy_dtheta / magnitude
        else:
            return 1.0, 0.0  # 默认方向
    
    def get_rod_direction(self, theta1, theta2):
        """
        计算连接两个节点的棍子方向（单位向量）
        从节点1指向节点2
        """
        x1, y1 = self.sim.get_spiral_xy(theta1)
        x2, y2 = self.sim.get_spiral_xy(theta2)
        
        dx = x2 - x1
        dy = y2 - y1
        
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 1e-10:
            return dx / magnitude, dy / magnitude
        else:
            return 1.0, 0.0  # 默认方向
    
    def calculate_angle_between_vectors(self, v1, v2):
        """
        计算两个向量之间的夹角（弧度）
        v1, v2: (x, y) 元组表示的向量
        """
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        # 限制dot_product在[-1, 1]范围内，避免数值误差
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return np.arccos(np.abs(dot_product))  # 取绝对值确保夹角在[0, π/2]
    
    def calculate_velocities(self):
        """
        计算所有节点的速度大小
        返回: 速度数组，单位 cm/s
        """
        n_nodes = self.sim.num_segments + 1
        velocities = np.zeros(n_nodes)
        
        # 龙头速度已知
        velocities[0] = self.head_speed
        
        # 逐个计算后续节点的速度
        for i in range(self.sim.num_segments):
            theta_current = self.sim.thetas[i]
            theta_next = self.sim.thetas[i + 1]
            
            # 计算当前节点的切线方向（速度方向）
            tangent_current = self.get_tangent_direction(theta_current)
            
            # 计算下一个节点的切线方向（速度方向）
            tangent_next = self.get_tangent_direction(theta_next)
            
            # 计算棍子方向（从当前节点指向下一个节点）
            rod_direction = self.get_rod_direction(theta_current, theta_next)
            
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

def compute_velocities(seconds: int = 300, output_path: Path | None = None):
    """
    计算 t=0..seconds (单位: 秒) 每一秒的节点速度，并写入 Excel。
    """
    # 创建仿真实例
    sim = DragonSimulation()
    sim.dt = 1.0  # 设置每秒一步
    
    # 创建速度计算器
    vel_calc = VelocityCalculator(sim)
    
    # 节点总数
    n_nodes = sim.num_segments + 1  # 0..223 共 224 个
    
    # 表头（列）为 0..seconds 秒
    columns = [f"{t} s" for t in range(0, seconds + 1)]
    
    # 行索引（节点速度）
    index_labels = ["龙头 (m/s)"]
    for i in range(1, n_nodes):
        index_labels.append(f"第{i}节龙身 (m/s)")
    
    # 数据矩阵：行数 = n_nodes，列数 = seconds + 1
    data = np.zeros((n_nodes, seconds + 1), dtype=float)
    
    def fill_column(col: int):
        """将当前时刻的所有节点速度（m/s）写入到第 col 列。"""
        velocities_cm_s = vel_calc.calculate_velocities()  # cm/s
        velocities_m_s = velocities_cm_s / 100.0  # 转换为 m/s
        
        for node in range(n_nodes):
            data[node, col] = velocities_m_s[node]
    
    # t = 0
    fill_column(0)
    
    # 按秒推进并记录
    for t in range(1, seconds + 1):
        sim.step()
        fill_column(t)
        if t % 10 == 0 or t == seconds:
            print(f"已完成 {t}/{seconds} 秒的速度计算")
    
    # 默认输出到项目根目录
    if output_path is None:
        project_root = Path(__file__).resolve().parents[1]
        output_path = project_root / "velocity_results.xlsx"
    
    # 写出 Excel（保留 6 位小数）
    data = np.round(data, 6)
    df = pd.DataFrame(data=data, index=index_labels, columns=columns)
    
    try:
        df.to_excel(output_path, sheet_name="velocities")
    except Exception:
        df.to_excel(output_path)
    
    print(f"速度计算结果已写入：{output_path}")
    return df
