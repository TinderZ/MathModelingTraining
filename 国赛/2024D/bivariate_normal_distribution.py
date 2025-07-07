import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# --- 参数定义 ---
# 均值
mean = 0
# 标准差
std_dev = 120

# --- 数据准备 ---
# 创建 x 和 y 轴的坐标点. 范围从 -3*标准差 到 3*标准差，以覆盖大部分分布区域.
x = np.linspace(-3 * std_dev, 3 * std_dev, 300)
y = np.linspace(-3 * std_dev, 3 * std_dev, 300)
# 将 x 和 y 坐标点组合成一个网格
X, Y = np.meshgrid(x, y)

# 将网格坐标 (X, Y) 组合成一个 (..., 2) 形状的数组，以供后续计算使用
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# --- 计算概率密度函数 (PDF) ---
# 定义协方差矩阵.
# 因为 x 和 y 是独立的, 所以它们之间的协方差为 0.
# 矩阵对角线上的元素是方差 (标准差的平方).，
cov_matrix = [[std_dev**2, 0], [0, std_dev**2]]

# 定义均值向量
mean_vector = [mean, mean]

# 创建一个二元正态分布对象
rv = multivariate_normal(mean_vector, cov_matrix)

# 计算每个 (X, Y) 点上的概率密度函数值
Z = rv.pdf(pos)

# --- 绘制3D图像 ---
# 创建一个图形实例
fig = plt.figure(figsize=(12, 8))
# 在图形中添加一个3D坐标轴
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面图, cmap='viridis' 设置了颜色映射
ax.plot_surface(X, Y, Z, cmap='viridis')

# --- 设置坐标轴和标题 ---
ax.set_xlabel('X 轴')
ax.set_ylabel('Y 轴')
ax.set_zlabel('概率密度 (PDF)')
ax.set_title('二元正态分布的3D概率密度函数')

# --- 显示图像 ---
plt.show() 