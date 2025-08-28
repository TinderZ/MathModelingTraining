#!/usr/bin/env python3
"""
测试修改后的热力图布局
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from velocity_heatmap_visualizer import VelocityHeatmapVisualizer
from t4simulate import DragonRectangleSimulation

def main():
    """测试新的热力图布局"""
    print("测试修改后的热力图布局...")
    
    # 创建仿真实例
    sim = DragonRectangleSimulation(pitch=170.0, debug=False)
    
    # 创建可视化器
    visualizer = VelocityHeatmapVisualizer(sim)
    
    # 测试时间点
    test_times = [14, 15, 16, 17]
    
    print(f"生成时间点 {test_times} 的热力图...")
    
    try:
        # 生成修改后的多时间点对比图
        fig, axes = visualizer.create_multi_time_heatmap(
            time_points=test_times,
            save_path="test_modified_layout.png"
        )
        
        print("新布局热力图生成成功！")
        print("文件保存为: test_modified_layout.png")
        print("特点:")
        print("- 标题位于图片下方")
        print("- 颜色条位于最右边")
        print("- 优化的间距和布局")
        
    except Exception as e:
        print(f"生成热力图时出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
