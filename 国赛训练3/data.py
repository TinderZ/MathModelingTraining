import pandas as pd

# 假设数据已复制到字符串中（或者从文件读取）
data = """
"""

# 将字符串数据按行分割并处理
lines = data.split('\n')
cleaned_data = [line.split() for line in lines]  # 按空格分割，自动删除多余空格

# 转换为DataFrame
df = pd.DataFrame(cleaned_data, columns=['Column1', 'Column2'])

# 保存为xlsx文件
df.to_excel('output.xlsx', index=False)

print("数据已保存到 output.xlsx")



import matplotlib.pyplot as plt
import pandas as pd

df['Column1'] = pd.to_numeric(df['Column1'])
df['Column2'] = pd.to_numeric(df['Column2'])

# 创建折线图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(df['Column1'], df['Column2'], marker='o', linestyle='-')  # 折线图，带点标记
plt.xlabel('X-axis (Column1)')  # x轴标签
plt.ylabel('Y-axis (Column2)')  # y轴标签
plt.title('')  # 图标题
plt.grid(True)  # 添加网格线
plt.show()