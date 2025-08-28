#!/usr/bin/env python3
"""
运行位置和速度计算的主脚本
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from position_velocity_calculator import compute_positions_and_velocities

def main():
    """主函数"""
    print("="*80)
    print("板凳龙位置和速度计算程序")
    print("时间范围：-100s 到 100s")
    print("="*80)
    
    try:
        # 运行计算
        position_df, velocity_df = compute_positions_and_velocities(
            start_time=-100, 
            end_time=100, 
            pitch=170.0,
            output_path=Path("dragon_trajectory_results.xlsx")
        )
        
        print("\n计算完成！")
        print(f"位置数据形状: {position_df.shape}")
        print(f"速度数据形状: {velocity_df.shape}")
        
        # 显示一些样本数据
        print("\n位置数据样本（前5行，前5列）:")
        print(position_df.iloc[:5, :5])
        
        print("\n速度数据样本（前5行，前5列）:")
        print(velocity_df.iloc[:5, :5])
        
    except Exception as e:
        print(f"计算过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
