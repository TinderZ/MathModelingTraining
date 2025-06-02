import pandas as pd
import numpy as np

# 读取数据
file_path = r'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\Huskies.xlsx'
df = pd.read_excel(file_path)

# 提取所有唯一的球员ID
players = pd.concat([df['OriginPlayerID'], df['DestinationPlayerID']]).unique()
player_to_index = {player: idx for idx, player in enumerate(players)}
num_players = len(players)

# 初始化邻接矩阵
W = np.zeros((num_players, num_players))

# 遍历数据，构建邻接矩阵
for index, row in df.iterrows():
    origin = player_to_index[row['OriginPlayerID']]
    destination = player_to_index[row['DestinationPlayerID']]
    if row['EventSubType'] == 'smart pass':
        weight = 3
    else:
        weight = 1
    W[origin][destination] += weight

# 转换为无向图
W = W + W.T

# 计算每个球员的度
degrees = np.sum(W, axis=1)







# 统计每个球员的出场次数
# 创建数据框来统计传球者的出场
origin_appearances = df[['OriginPlayerID', 'MatchID']].copy()
origin_appearances.rename(columns={'OriginPlayerID': 'PlayerID'}, inplace=True)
# 创建数据框来统计接球者的出场
destination_appearances = df[['DestinationPlayerID', 'MatchID']].copy()
destination_appearances.rename(columns={'DestinationPlayerID': 'PlayerID'}, inplace=True)
# 合并出场数据
all_appearances = pd.concat([origin_appearances, destination_appearances])
# 去除重复的MatchID和PlayerID组合
unique_appearances = all_appearances.drop_duplicates(subset=['PlayerID', 'MatchID'])
# 统计每个球员的出场次数
player_appearances = unique_appearances['PlayerID'].value_counts().reset_index()
player_appearances.columns = ['PlayerID', 'AppearanceCount']

# 创建结果数据框
result_df = pd.DataFrame({
    'PlayerID': players,
    'Degree': degrees,
    #'AverageX': avg_coords.set_index('PlayerID')['x'],
    #'AverageY': avg_coords.set_index('PlayerID')['y'],
    'AppearanceCount': player_appearances.set_index('PlayerID')['AppearanceCount']
})

# 处理可能的缺失值
result_df['AverageX'].fillna(0, inplace=True)
result_df['AverageY'].fillna(0, inplace=True)
result_df['AppearanceCount'].fillna(0, inplace=True)

# 保存结果到Excel
output_path = r'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\Husk_player.xlsx'
result_df.to_excel(output_path, index=False)

# # 保存邻接矩阵到Excel
# adjacency_df = pd.DataFrame(W, index=players, columns=players)
# adjacency_output_path = r'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\AdjacencyMatrix.xlsx'
# adjacency_df.to_excel(adjacency_output_path)















