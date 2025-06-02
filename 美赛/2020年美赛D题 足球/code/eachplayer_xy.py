import pandas as pd

# 读取数据
file_path = r"E:\数模\训练题\2020年美赛D题 足球\data\passingevents\eachcoach\coach3.xlsx"
df = pd.read_excel(file_path)

# 提取所有唯一的球员ID
players = pd.concat([df['Source'], df['Target']]).unique()

# 提取传球者的坐标，并重命名列
origin_coords = df[['Source', 'x1', 'y1']].copy()
origin_coords.rename(columns={'Source': 'Id', 'x1': 'x', 'y1': 'y'}, inplace=True)

# 提取接球者的坐标，并重命名列
destination_coords = df[['Target', 'x2', 'y2']].copy()
destination_coords.rename(columns={'Target': 'Id', 'x2': 'x', 'y2': 'y'}, inplace=True)

# 合并传球者和接球者的坐标数据
all_coords = pd.concat([origin_coords, destination_coords])

# 按PlayerID分组，计算平均x和y坐标
avg_coords = all_coords.groupby('Id').mean().reset_index()

# 创建一个包含所有球员ID的DataFrame
all_players_df = pd.DataFrame({'Id': players})

# 左连接平均坐标数据
result_df = all_players_df.merge(avg_coords, on='Id', how='left')

# 填充缺失值为0
result_df['x'].fillna(0, inplace=True)
result_df['y'].fillna(0, inplace=True)

# 重命名列
#result_df.rename(columns={'x': 'AverageX', 'y': 'AverageY'}, inplace=True)

# 保存结果到Excel
output_path = r"E:\数模\训练题\2020年美赛D题 足球\data\passingevents\eachcoach\coach3_node.xlsx"
result_df.to_excel(output_path, index=False)