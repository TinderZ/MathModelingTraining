import pandas as pd
import numpy as np
import os

# 定义基础路径
base_path = r'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\flexibility'

# 循环处理每个文件
for i in range(1, 39):
    for team in ['H', 'O']:
        # 生成文件名
        filename = f"{i}{team}_flex.xlsx"
        file_path = os.path.join(base_path, filename)
        
        # 检查文件是否存在
        if os.path.exists(file_path):
            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            
            # 计算下一个质心的x和y坐标
            df['next_centroid_x'] = df['centroid_x'].shift(-1)
            df['next_centroid_y'] = df['centroid_y'].shift(-1)
            
            # 计算欧几里得距离
            df['distance'] = np.sqrt((df['centroid_x'] - df['next_centroid_x'])**2 + (df['centroid_y'] - df['next_centroid_y'])**2)
            
            # 舍去最后一个是NaN的行
            df = df[:-1]
            
            # 计算所有距离的平均值
            average_distance = df['distance'].mean()
            
            # 保存回Excel文件
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name='Sheet1', index=False)
                worksheet = writer.sheets['Sheet1']
                last_row = len(df) + 1
                worksheet.cell(row=last_row + 1, column=3, value='Average Distance')
                worksheet.cell(row=last_row + 1, column=4, value=average_distance)
        else:
            print(f"文件{filename}不存在，跳过。")