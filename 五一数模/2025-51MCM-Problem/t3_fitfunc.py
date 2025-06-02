import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('待拟合函数.xlsx', sheet_name='Sheet1')

# 创建特征
df['I'] = df['x'].apply(lambda x: 1 if 13 <= x <= 17 else 0)
df['x_squared'] = df['x']**2
df['x_3'] = df['x']**3
df['I_x'] = df['I'] * df['x']
df['I_intercept'] = df['I']  # 线性修正的截距项

# 构建特征矩阵和标签
X = df[['x', 'x_squared', 'x_3', 'I_intercept', 'I_x']]
y = df['y']

# 拟合模型
model = LinearRegression()
model.fit(X, y)

# 提取系数
a0 = model.intercept_
a1, a2, a3, b0, b1 = model.coef_

# 定义分段函数
def fitted_func(x):
    base = a0 + a1*x + a2*x**2 + a3*x**3
    if 13 <= x <= 17:
        return base + b0 + b1*x
    else:
        return base

# 预测并绘图
x_values = np.linspace(8, 24, 200)
y_pred = [fitted_func(x) for x in x_values]

plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], color='red', label='实际数据')
plt.plot(x_values, y_pred, label='拟合曲线', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('function_fit-together.png')


# 输出拟合公式
print(f"全局多项式公式: {a0:.4f} + {a1:.4f}x + {a2:.4f}x² + {a3:.4f}x³")
print(f"修正项公式 (13≤x≤17): + ({b0:.4f} + {b1:.4f}x)")
print(f"完整公式:")
print("当 x <13 或 x>17 时: y =", f"{a0:.4f} + {a1:.4f}x + {a2:.4f}x² + {a3:.4f}x³")
print("当 13≤x≤17 时: y =", f"{a0:.4f} + {a1:.4f}x + {a2:.4f}x² + {a3:.4f}x³ + {b0:.4f} + {b1:.4f}x")

# 计算预测值
y_pred = [fitted_func(x) for x in df['x']]  # 使用自定义函数计算预测值

# 计算各项误差指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(df['y'], y_pred)
mae = mean_absolute_error(df['y'], y_pred)
r2 = r2_score(df['y'], y_pred)

# 打印误差指标
print("\n误差分析报告：")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"R²决定系数: {r2:.4f}")

# 计算并打印每个点的绝对误差
df['绝对误差'] = np.abs(df['y'] - y_pred)
print("\n各点误差详情：")
print(df[['x', 'y', '绝对误差']])

# 绘制残差图
plt.figure(figsize=(10, 4))
plt.scatter(df['x'], df['y'] - y_pred, color='blue', s=50, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('残差图')
plt.xlabel('x')
plt.ylabel('残差 (实际值 - 预测值)')
plt.grid(True)
plt.tight_layout()
plt.savefig('residual_plot.png')
plt.show()