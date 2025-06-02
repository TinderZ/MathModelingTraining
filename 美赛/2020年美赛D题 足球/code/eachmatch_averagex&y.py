import pandas as pd
import os

# 输入文件夹路径
input_folder = r"E:\数模\训练题\2020年美赛D题 足球\data\passingevents\eachmatch"
# 输出文件夹路径
output_folder_H_node = r"E:\数模\训练题\2020年美赛D题 足球\data\passingevents\_eachmatch_\{}_H_node.xlsx"
output_folder_O_node = r"E:\数模\训练题\2020年美赛D题 足球\data\passingevents\_eachmatch_\{}_O_node.xlsx"
output_folder_H_edge = r"E:\数模\训练题\2020年美赛D题 足球\data\passingevents\_eachmatch_\{}_H_edge.xlsx"
output_folder_O_edge = r"E:\数模\训练题\2020年美赛D题 足球\data\passingevents\_eachmatch_\{}_O_edge.xlsx"

for x in range(1, 39):
    # 构建输入文件路径
    file_path = os.path.join(input_folder, f"{x}.xlsx")
    
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 处理Huskies球员的平均坐标
    huskies_df = df[df['TeamID'] == 'Huskies']
    
    # 获取所有Huskies球员的唯一列表
    huskies_players = list(set(huskies_df['OriginPlayerID'].tolist() + huskies_df['DestinationPlayerID'].tolist()))
    
    # 创建一个空列表来存储数据
    data_H = []
    
    for player in huskies_players:
        # 获取球员作为传球者和接球者的坐标
        origin_coords = huskies_df[huskies_df['OriginPlayerID'] == player][['x1', 'y1']]
        destination_coords = huskies_df[huskies_df['DestinationPlayerID'] == player][['x2', 'y2']]
        origin_coords.rename(columns={'x1': 'x', 'y1': 'y'}, inplace=True)
        destination_coords.rename(columns={'x2': 'x', 'y2': 'y'}, inplace=True)
        # 合并坐标
        all_coords = pd.concat([origin_coords, destination_coords])
        
        # 计算平均值
        avg_x = all_coords['x'].mean()
        avg_y = all_coords['y'].mean()
        
        # 将数据添加到列表中
        data_H.append({'Id': player, 'Label': player, 'x': avg_x, 'y': avg_y})
    
    # 创建 DataFrame
    avg_coords_H = pd.DataFrame(data_H)
    
    # 保存Huskies球员的平均坐标到Excel
    output_path_H_node = output_folder_H_node.format(x)
    avg_coords_H.to_excel(output_path_H_node, index=False)

    #-------------------------------------------------------------------------------------------------------------

    # 处理对手球员的平均坐标
    opponent_df = df[df['TeamID'] != 'Huskies']
    
    # 获取所有对手球员的唯一列表
    opponent_players = list(set(opponent_df['OriginPlayerID'].tolist() + opponent_df['DestinationPlayerID'].tolist()))
    
    # 创建一个空列表来存储数据
    data_O = []
    
    for player in opponent_players:
        # 获取球员作为传球者和接球者的坐标
        origin_coords = opponent_df[opponent_df['OriginPlayerID'] == player][['x1', 'y1']]
        destination_coords = opponent_df[opponent_df['DestinationPlayerID'] == player][['x2', 'y2']]
        origin_coords.rename(columns={'x1': 'x', 'y1': 'y'}, inplace=True)
        destination_coords.rename(columns={'x2': 'x', 'y2': 'y'}, inplace=True)

        # 合并坐标
        all_coords = pd.concat([origin_coords, destination_coords])
        
        # 计算平均值
        avg_x = all_coords['x'].mean()
        avg_y = all_coords['y'].mean()
        
        # 将数据添加到列表中
        data_O.append({'Id': player, 'Label': player, 'x': avg_x, 'y': avg_y})
    
    # 创建 DataFrame
    avg_coords_O = pd.DataFrame(data_O)
    
    # 保存对手球员的平均坐标到Excel
    output_path_O_node = output_folder_O_node.format(x)
    avg_coords_O.to_excel(output_path_O_node, index=False)
    
    # 提取Huskies的边数据（A到G列）
    huskies_edge = huskies_df.iloc[:, :11]  # A到G列是前7列
    
    huskies_edge = huskies_edge.rename(columns={
        'OriginPlayerID': 'Source',
        'DestinationPlayerID': 'Target'
    })

    # 保存到Excel
    output_path_H_edge = output_folder_H_edge.format(x)
    huskies_edge.to_excel(output_path_H_edge, index=False)
    
    # 提取对手的边数据（A到G列）
    opponent_edge = opponent_df.iloc[:, :11]
    
    # 保存到Excel
    output_path_O_edge = output_folder_O_edge.format(x)
    opponent_edge.to_excel(output_path_O_edge, index=False)





