import numpy as np
import pandas as pd
from pathlib import Path
from t1simulate import DragonSimulation
from velocity_calculator import VelocityCalculator

def compute_positions_and_velocities(seconds: int = 300, 
                                   pos_output_path: Path | None = None,
                                   vel_output_path: Path | None = None):
    """
    同时计算位置和速度，并分别保存到不同的Excel文件中
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
    
    # 位置数据：行索引（节点 -> x/y 两行）
    pos_index_labels = ["龙头x(m)", "龙头y(m)"]
    for i in range(1, n_nodes):
        pos_index_labels.append(f"第{i}节龙身x(m)")
        pos_index_labels.append(f"第{i}节龙身y(m)")
    
    # 速度数据：行索引
    vel_index_labels = ["龙头 (m/s)"]
    for i in range(1, n_nodes):
        vel_index_labels.append(f"第{i}节龙身 (m/s)")
    
    # 数据矩阵
    pos_data = np.zeros((2 * n_nodes, seconds + 1), dtype=float)
    vel_data = np.zeros((n_nodes, seconds + 1), dtype=float)
    
    def fill_columns(col: int):
        """填充位置和速度数据"""
        # 位置数据
        x_cm, y_cm = sim.get_all_node_coords()
        x_m = x_cm / 100.0
        y_m = y_cm / 100.0
        
        for node in range(n_nodes):
            pos_data[2 * node, col] = x_m[node]
            pos_data[2 * node + 1, col] = y_m[node]
        
        # 速度数据
        velocities_cm_s = vel_calc.calculate_velocities()
        velocities_m_s = velocities_cm_s / 100.0
        
        for node in range(n_nodes):
            vel_data[node, col] = velocities_m_s[node]
    
    # t = 0
    fill_columns(0)
    
    # 按秒推进并记录
    for t in range(1, seconds + 1):
        sim.step()
        fill_columns(t)
        if t % 10 == 0 or t == seconds:
            print(f"已完成 {t}/{seconds} 秒的计算")
    
    # 设置默认输出路径
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "result_t1.xlsx"
    
    # 保存位置数据和速度数据到同一文件的不同sheet
    pos_data = np.round(pos_data, 6)
    pos_df = pd.DataFrame(data=pos_data, index=pos_index_labels, columns=columns)
    
    vel_data = np.round(vel_data, 6)
    vel_df = pd.DataFrame(data=vel_data, index=vel_index_labels, columns=columns)
    
    # 使用ExcelWriter同时写入多个sheet
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pos_df.to_excel(writer, sheet_name='positions')
        vel_df.to_excel(writer, sheet_name='velocities')
    
    print(f"位置和速度数据已写入：{output_path}")
    
    return pos_df, vel_df

if __name__ == "__main__":
    # 运行计算：0..300 秒
    compute_positions_and_velocities(seconds=300)