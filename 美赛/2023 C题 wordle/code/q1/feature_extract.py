import pandas as pd
import numpy as np
import holidays

# 读取数据
df = pd.read_excel('q1_timeseries.xlsx', sheet_name='Sheet1')

df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')

# 基于已知的 2022 年 7 月 1 日是星期五，来计算所有日期的星期几（星期几从 1 开始）
known_date = pd.Timestamp('2022-01-07')
known_weekday = 5  # 1 表示周一，5 表示周五

# 重新计算每个日期的星期几，范围从 1（周一）到 7（周日）
df['day of the week'] = df['Date'].apply(lambda date: ((known_weekday - 1) + (date - known_date).days) % 7 + 1)

for i in range(1, 8):
    df[f'Day_{i}'] = df['day of the week'].apply(lambda x: 1 if x == i else 0)

# 如果你不想在原始 DataFrame 中保留 '星期几' 列，可以选择删除它
df = df.drop('day of the week', axis=1)

# 获取美国节假日
us_holidays = holidays.US()
df.set_index('Date', inplace=True) 
# 判断是否是节假日
df['IsHoliday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)


# 创建滞后特征（1到4天）
for i in range(1, 8):
    df[f'Lag_{i}'] = df['Number of reported results'].shift(i)


# 选择特征和目标
features = ['Contest number', 'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7', 
            'IsHoliday', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7']
target = 'Number of reported results'

# 填充滞后特征中的NaN值
df[features] = df[features].fillna(0)


# 选择要保存的列（特征和目标）
data_to_save = df[features + [target]]

# 将数据保存到 Excel 文件
output_filename = 'feature_extracted_data.xlsx'
data_to_save.to_excel(output_filename, index=True)

print(f"特征提取的数据已保存至 {output_filename}")