#!/usr/bin/env python3
"""
测试新的颜色条范围设置
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from velocity_heatmap_visualizer import VelocityHeatmapVisualizer
from t4simulate import DragonRectangleSimulation

def main():
    """测试新的颜色条范围"""
    print("测试新的颜色条范围设置...")
    
    # 创建仿真实例
    sim = DragonRectangleSimulation(pitch=170.0, debug=False)
    
    # 创建可视化器
    visualizer = VelocityHeatmapVisualizer(sim)
    
    # 测试时间点
    test_times = [14, 15, 16, 17]
    
    print(f"生成时间点 {test_times} 的热力图...")
    print("颜色条设置:")
    print("- 范围: [global_vmin*0.8, global_vmax/0.8]")
    print("- 标签: 1.0, 1.2, 1.4 m/s")
    
    try:
        # 生成新颜色条设置的热力图
        fig, axes = visualizer.create_multi_time_heatmap(
            time_points=test_times,
            save_path="test_new_colorbar.png"
        )
        
        print("新颜色条热力图生成成功！")
        print("文件保存为: test_new_colorbar.png")
        
    except Exception as e:
        print(f"生成热力图时出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
