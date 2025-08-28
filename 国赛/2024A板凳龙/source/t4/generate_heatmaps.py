#!/usr/bin/env python3
"""
生成速度热力图的简单脚本
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from velocity_heatmap_visualizer import VelocityHeatmapVisualizer
from t4simulate import DragonRectangleSimulation

def main():
    """主函数"""
    print("开始生成板凳龙速度热力图...")
    
    # 创建仿真实例
    sim = DragonRectangleSimulation(pitch=170.0, debug=False)
    
    # 创建可视化器
    visualizer = VelocityHeatmapVisualizer(sim)
    
    # 生成指定时间点的热力图
    test_times = [-100, -50, 0, 50, 100]
    
    print(f"生成时间点 {test_times} 的热力图...")
    
    try:
        # 生成多时间点对比图
        fig, axes = visualizer.create_multi_time_heatmap(
            time_points=test_times,
            save_path="dragon_velocity_heatmap_comparison.png"
        )
        
        print("多时间点热力图生成成功！")
        print("文件保存为: dragon_velocity_heatmap_comparison.png")
        
        # 可选：生成单个详细图
        print("\n生成 t=0s 的详细热力图...")
        visualizer.create_single_large_heatmap(
            time_point=0,
            save_path="dragon_velocity_detailed_t0s.png"
        )
        
        print("详细热力图生成成功！")
        print("文件保存为: dragon_velocity_detailed_t0s.png")
        
    except Exception as e:
        print(f"生成热力图时出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
