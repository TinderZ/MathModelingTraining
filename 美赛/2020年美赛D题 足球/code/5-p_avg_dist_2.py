import pandas as pd
import numpy as np
import os

# 定义基础路径
base_path = r'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\flexibility'

# 创建一个列表来存储比赛序号、H队平均距离和O队平均距离
data = []

for match_number in range(1, 39):  # 比赛编号从1到38
    # 生成文件名
    if match_number < 10:
        h_filename = f"{match_number}H_flex.xlsx"
        o_filename = f"{match_number}O_flex.xlsx"
    else:
        h_filename = f"{match_number}H_flex.xlsx"
        o_filename = f"{match_number}O_flex.xlsx"
    
    h_file_path = os.path.join(base_path, h_filename)
    o_file_path = os.path.join(base_path, o_filename)
    
    # 初始化平均距离
    h_average = np.nan
    o_average = np.nan
    
    # 处理H队文件
    if os.path.exists(h_file_path):
        df_h = pd.read_excel(h_file_path, sheet_name='Sheet1')
        df_h['next_centroid_x'] = df_h['centroid_x'].shift(-1)
        df_h['next_centroid_y'] = df_h['centroid_y'].shift(-1)
        df_h['distance'] = np.sqrt((df_h['centroid_x'] - df_h['next_centroid_x'])**2 + (df_h['centroid_y'] - df_h['next_centroid_y'])**2)
        df_h = df_h[:-1]
        h_average = df_h['distance'].mean()
    
    # 处理O队文件
    if os.path.exists(o_file_path):
        df_o = pd.read_excel(o_file_path, sheet_name='Sheet1')
        df_o['next_centroid_x'] = df_o['centroid_x'].shift(-1)
        df_o['next_centroid_y'] = df_o['centroid_y'].shift(-1)
        df_o['distance'] = np.sqrt((df_o['centroid_x'] - df_o['next_centroid_x'])**2 + (df_o['centroid_y'] - df_o['next_centroid_y'])**2)
        df_o = df_o[:-1]
        o_average = df_o['distance'].mean()
    
    # 将数据添加到列表中
    data.append([match_number, h_average, o_average])

# 创建DataFrame
df_result = pd.DataFrame(data, columns=['比赛序号', 'H队平均距离', 'O队平均距离'])

# 保存到Excel文件
output_path = r'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\flexibility.xlsx'
df_result.to_excel(output_path, index=False)