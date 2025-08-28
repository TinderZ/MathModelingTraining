import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 将 t1 目录添加到 Python 路径中，以便导入其模块
t1_path = Path(__file__).resolve().parents[1] / 't1'
sys.path.append(str(t1_path))

from t1simulate import DragonSimulation
from velocity_calculator import VelocityCalculator

def compute_state_at_specific_time(target_time: float, output_path: Path):
    """
    计算在特定时间点的所有节点的位置和速度，并保存到Excel。

    Args:
        target_time (float): 目标计算时间 (秒)
        output_path (Path): Excel 输出文件的路径
    """
    # 1. 初始化仿真
    # 构造函数会进行初始状态的计算，这需要一些时间
    print("正在初始化仿真，请稍候...")
    sim = DragonSimulation()
    print("仿真初始化完成。")

    # 2. 运行仿真直到目标时间
    # 为了精确到达 target_time，我们将时间步分为整数部分和小数部分
    
    # 先以 dt=1.0s 运行整数秒
    full_seconds = int(target_time)
    if full_seconds > 0:
        sim.dt = 1.0
        print(f"正在模拟 {full_seconds} 秒...")
        for t in range(1, full_seconds + 1):
            sim.step()
            if t % 20 == 0:
                print(f"  ...已完成 {t}/{full_seconds} 秒")
    
    # 再运行剩余的不足1秒的时间
    remaining_time = target_time - full_seconds
    if remaining_time > 1e-9: # 仅当剩余时间足够大时才步进
        print(f"正在模拟剩余的 {remaining_time:.4f} 秒...")
        sim.dt = remaining_time
        sim.step()

    print(f"仿真已到达目标时间: {sim.time:.4f}s")

    # 3. 在当前状态下计算位置和速度
    print("正在计算最终时刻的位置和速度...")
    vel_calc = VelocityCalculator(sim)
    
    n_nodes = sim.num_segments + 1
    
    # 获取位置 (单位: m)
    x_cm, y_cm = sim.get_all_node_coords()
    x_m = x_cm / 100.0
    y_m = y_cm / 100.0
    
    # 获取速度 (单位: m/s)
    velocities_cm_s = vel_calc.calculate_velocities()
    velocities_m_s = velocities_cm_s / 100.0
    
    # 4. 整理数据并存入 DataFrame
    node_labels = ["龙头"] + [f"第{i}节龙身" for i in range(1, n_nodes)]
    
    data = {
        'x坐标(m)': np.round(x_m, 6),
        'y坐标(m)': np.round(y_m, 6),
        '速度(m/s)': np.round(velocities_m_s, 6)
    }
    
    df = pd.DataFrame(data, index=node_labels)
    df.index.name = "节点"

    # 5. 保存到 Excel 文件
    try:
        df.to_excel(output_path, sheet_name=f't={target_time}s_data')
        print(f"计算结果已成功写入到: {output_path}")
    except Exception as e:
        print(f"写入 Excel 文件时出错: {e}")

if __name__ == "__main__":
    # --- 参数设置 ---
    TARGET_TIME = 412.4739  # 目标时间 (秒)
    
    # --- 文件路径设置 ---
    # 将输出文件保存到项目的根目录
    project_root = Path(__file__).resolve().parents[2]
    output_file = project_root / "result2.xlsx"
    
    # --- 执行计算 ---
    compute_state_at_specific_time(TARGET_TIME, output_file)