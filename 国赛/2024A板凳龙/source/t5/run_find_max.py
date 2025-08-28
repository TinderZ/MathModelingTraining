#!/usr/bin/env python3
"""
运行速度最大值搜索的脚本
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from find_v_max import main

if __name__ == "__main__":
    print("开始搜索速度最大的点...")
    main()
