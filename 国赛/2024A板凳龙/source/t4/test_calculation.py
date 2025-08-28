#!/usr/bin/env python3
"""
测试位置和速度计算
"""

import sys
from pathlib import Path
import numpy as np

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from position_velocity_calculator import PositionVelocityCalculator
from t4simulate import DragonRectangleSimulation

def main():
    """测试计算逻辑"""
    print("测试位置和速度计算...")
    
    # 创建仿真实例
    sim = DragonRectangleSimulation(pitch=170.0, debug=False)
    
    # 创建计算器
    calc = PositionVelocityCalculator(sim)
    
    print(f"节点数量: {len(sim.positions)}")
    print(f"节点相位: {len(sim.node_phases)}")
    
    # 测试几个时间点
    test_times = [-100, -50, 0, 50, 100]
    
    for t in test_times:
        sim.time = t
        sim.update_all_positions()
        
        print(f"\n时间 t={t}s:")
        print(f"  龙头位置: ({sim.positions[0][0]:.2f}, {sim.positions[0][1]:.2f}) cm")
        print(f"  龙头相位: {sim.node_phases[0]}")
        
        # 计算速度
        try:
            velocities = calc.calculate_velocities_at_time()
            print(f"  龙头速度: {velocities[0]:.2f} cm/s")
            print(f"  第10节速度: {velocities[10]:.2f} cm/s")
            print(f"  龙尾速度: {velocities[-1]:.2f} cm/s")
        except Exception as e:
            print(f"  速度计算出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
