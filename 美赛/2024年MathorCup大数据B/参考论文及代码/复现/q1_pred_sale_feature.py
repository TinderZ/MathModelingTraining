# 提取销量特征的完整代码
import pandas as pd
import numpy as np
from datetime import timedelta
import random

# 假设销量数据文件为 "sales_data.csv"，包含列 ["品类", "日期", "销量"]
file_path = "附件2.csv"

# 读取销量数据
# 由于文件中可能包含中文列名，使用 utf-8 编码读取
try:
    df_sales = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df_sales = pd.read_csv(file_path, encoding='gbk')

# 将日期列转换为 pandas 日期类型
df_sales['日期'] = pd.to_datetime(df_sales['日期'], format='%Y/%m/%d')

# -------------------------------------------
# 基于已知的 2022 年 7 月 1 日是星期五，来计算所有日期的星期几（星期几从 1 开始）
known_date = pd.Timestamp('2022-07-01')
known_weekday = 5  # 1 表示周一，5 表示周五

# 重新计算每个日期的星期几，范围从 1（周一）到 7（周日）
df_sales['星期几'] = df_sales['日期'].apply(lambda date: ((known_weekday - 1) + (date - known_date).days) % 7 + 1)
# 计算是否周末（6 和 7 表示周六和周日）
# df_sales['是否周末'] = df_sales['星期几'].apply(lambda x: 1 if x in [6, 7] else 0)

for i in range(1, 8):
    df_sales[f'星期_{i}'] = df_sales['星期几'].apply(lambda x: 1 if x == i else 0)

# 如果你不想在原始 DataFrame 中保留 '星期几' 列，可以选择删除它
df_sales = df_sales.drop('星期几', axis=1)
# -------------------------------------------

# 手动加入一些常见的法定节假日标注，仅标注现有数据范围内的节假日
holidays_manual = [
    pd.Timestamp('2023-04-05'),  # 清明节
    pd.Timestamp('2023-05-01'),  # 劳动节
    pd.Timestamp('2023-05-14'),  # 母亲节
    pd.Timestamp('2023-05-20'),
    pd.Timestamp('2023-05-21'),
    pd.Timestamp('2023-06-01'),  # 儿童节
    pd.Timestamp('2023-06-18'),  # 父亲节
    pd.Timestamp('2023-06-22'),  # 端午节
    pd.Timestamp('2022-08-04'),  # 七夕情人节
    pd.Timestamp('2023-08-22'),  # 七夕情人节
    pd.Timestamp('2022-09-10'),  # 中秋节
    pd.Timestamp('2023-09-29'),  # 中秋节
]

df_sales['是否节假日'] = df_sales['日期'].apply(lambda date: 1 if date in holidays_manual else 0)

# 手动加入一些常见的促销日，仅标注现有数据范围内
promotions = (
        list(pd.date_range(start=pd.Timestamp('2023-06-01'), end=pd.Timestamp('2023-06-18'))) +
        [pd.Timestamp('2022-08-08'), pd.Timestamp('2022-08-18'),  # 8 8
         pd.Timestamp('2023-08-08'), pd.Timestamp('2023-08-18'),
         pd.Timestamp('2022-09-09'), pd.Timestamp('2023-09-09'),  # 99 大促
         ])
df_sales['是否促销日'] = df_sales['日期'].apply(lambda date: 1 if date in promotions else 0)


# 为所有品类构造滞后特征（例如，使用前 7 天的销量来预测当前销量）
for lag in range(1, 8):
    df_sales[f'滞后_{lag}天'] = df_sales.groupby('品类')['销量'].shift(lag)

# 去除由于构造滞后特征而导致缺失值的行
df_sales = df_sales.dropna()


# # 准备最终包含所有特征的特征表格
# feature_columns = ['日期', '品类', '销量', '是否周末', '是否节假日', '是否促销日'] + [
#     f'滞后_{lag}天' for lag in range(1, 8)] + ['星期_' + str(i) for i in range(1, 8)]
#
# # 最终特征数据表格
# feature_df = df_sales[feature_columns]
feature_df = df_sales
pd.set_option('display.max_columns', None)  # 显示所有列
print(feature_df.head())

# 保存特征表格到文件
feature_file_path = "销量预测特征表.csv"
feature_df.to_csv(feature_file_path, index=False)

print(f"特征提取完成，已保存至 {feature_file_path}")
