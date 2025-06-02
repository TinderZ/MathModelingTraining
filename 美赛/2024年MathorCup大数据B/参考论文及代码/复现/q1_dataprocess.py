import pandas as pd
import matplotlib.pyplot as plt
import random


# 读取CSV文件
file_path_1 = '附件1.csv'
file_path_2 = '附件2.csv'

try:
    df_inventory = pd.read_csv(file_path_1, encoding='utf-8')
    df_sales = pd.read_csv(file_path_2, encoding='utf-8')
except UnicodeDecodeError:
    # 若 utf-8 编码失败，尝试使用 gbk 编码读取
    df_inventory = pd.read_csv(file_path_1, encoding='gbk')
    df_sales = pd.read_csv(file_path_2, encoding='gbk')

# 查看前几行数据以了解数据的基本结构
df_inventory_head = df_inventory.head()
df_sales_head = df_sales.head()

# 显示数据表头信息
# print("数据表头信息：",df_inventory_head)
# print("数据表头信息：",df_sales_head)


# 查找库存量数据和销量数据中的缺失值情况
inventory_missing = df_inventory.isnull().sum()
sales_missing = df_sales.isnull().sum()

# 显示缺失值情况
# print("库存量数据缺失值情况：\n", inventory_missing,'\n', sales_missing)

# 随机选择多个唯一品类
selected_categories = random.sample(list(df_inventory['品类'].unique()), 4)

# 按月份对数据进行排序
df_inventory['月份'] = pd.to_datetime(df_inventory['月份'], format='%Y/%m月')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 使得坐标轴负号显示正常
# 设置图形大小
plt.figure(figsize=(12, 6))

# 绘制折线图
for category in selected_categories:
    category_data = df_inventory[df_inventory['品类'] == category]
    plt.plot(category_data['月份'], category_data['库存量'], label=f'{category}')

# 设置图形标签和标题
plt.xlabel('时间')
plt.ylabel('库存量')
plt.title('库存时间序列')
# 设置图例
plt.grid(True)
plt.legend(loc='upper left')
# 设置 X 轴标签旋转角度
plt.xticks(rotation=45)
# 显示图形
plt.show()





# 对日期列进行数据类型转换
df_sales['日期'] = pd.to_datetime(df_sales['日期'])

# 设置图形大小
plt.figure(figsize=(12, 6))

# 随机选择多个唯一品类并将其转换为列表
selected_categories = random.sample(df_sales['品类'].unique().tolist(), 5)

# 绘制折线图
for category in selected_categories:
    category_data = df_sales[df_sales['品类'] == category]
    plt.plot(category_data['日期'], category_data['销量'], label=f'{category}')

# 设置标签和标题
plt.xlabel('日期')
plt.xticks(rotation=45)
plt.ylabel('销量')
plt.title('不同品类销量时间变化趋势')

# 设置图例
plt.legend()

# 显示图形
plt.show()


