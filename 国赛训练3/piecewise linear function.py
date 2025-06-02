'''
Author: Zhurun Zhang
Date: 2025-04-13 17:39:42
LastEditors: Zhurun Zhang
LastEditTime: 2025-04-18 15:12:10
FilePath: \exp2e:\MathematicalModeling\训练题\国赛训练3\piecewise linear function.py
Description: Always happy to chat! Reach out via email < b23042510@njupt.edu.cn or 2857895300@qq.com >

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('数据.xlsx', header=None)  # 假设文件名为"数据.xlsx"，数据在第一列无表头
y = df[0].values  # 获取y值数组

# 生成x轴数据
n = len(y)  # 数据点数
x = np.linspace(6.4, 14, n)  # 从6.5到14，生成n个点，步长自动计算
dx = x[1] - x[0]  # 步长，应约为0.0001

# 绘制图形
plt.plot(x, y, '-', linewidth=1)  # 使用折线图，便于观察分段线性特征
plt.xlabel('x')
plt.ylabel('y')
plt.title('数据图')
plt.grid(True)  # 添加网格线，便于观察
plt.show()

# 计算一阶差分和二阶差分
dy = np.diff(y)  # 一阶差分，dy[i] = y[i+1] - y[i]
ddy = np.diff(dy)  # 二阶差分，ddy[i] = dy[i+1] - dy[i]

# 设置容差，检测斜率变化
tolerance = 1e-6  # 根据数据精度选择容差

# 找到分段点的索引
break_indices = [0]  # 起始点
for i in range(len(ddy)):
    if abs(ddy[i]) > tolerance:  # 二阶差分超过容差，表明斜率变化
        break_indices.append(i + 1)  # 分段点在i+1处
if break_indices[-1] != n - 1:  # 确保末点包含在内
    break_indices.append(n - 1)




# 去重并排序（以防万一）
break_indices = sorted(set(break_indices))


# 设置最小区间长度阈值
min_interval_length = 0.01  # 可根据需要调整，例如 0.01

# 筛选有效分段点
valid_break_indices = [break_indices[0]]  # 保留起始点
for i in range(1, len(break_indices)):
    if x[break_indices[i]] - x[valid_break_indices[-1]] >= min_interval_length:
        valid_break_indices.append(break_indices[i])

# 判断是否为分段线性函数
if len(valid_break_indices) > 2:
    print("数据是分段线性函数。")
else:
    print("数据不是分段线性函数，可能为单一线性函数或非线性函数。")

# 输出分段点
print("\n分段点（x 值）：")
for idx in valid_break_indices:
    print(f"{x[idx]:.5f}")

# 计算并输出各区间的线性函数公式
print("\n各分段的线性函数公式：")
for j in range(len(valid_break_indices) - 1):
    a = valid_break_indices[j]    # 区间起点索引
    b = valid_break_indices[j + 1]  # 区间终点索引
    if b > a:  # 确保区间内有多个点
        m = (y[b] - y[a]) / (x[b] - x[a])  # 计算斜率
        c = y[a] - m * x[a]  # 计算截距
        print(f"区间 [{x[a]:.5f}, {x[b]:.5f}]：y = {m:.6f} * x + {c:.6f}")
    else:
        print(f"区间 [{x[a]:.5f}, {x[b]:.5f}]：无有效线性函数（单点）")