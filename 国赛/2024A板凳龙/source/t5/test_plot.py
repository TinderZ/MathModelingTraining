#!/usr/bin/env python3
"""
测试绘图功能的简单脚本
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from find_v_max import main

if __name__ == "__main__":
    print("开始运行速度搜索和可视化...")
    print("这将生成:")
    print("1. Excel数据文件: max_velocity_search_results.xlsx")
    print("2. 时间-速度曲线图: velocity_time_curves.png")
    print()
    main()
