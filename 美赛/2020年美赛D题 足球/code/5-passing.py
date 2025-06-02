import pandas as pd
import os

def process_file(file_path, output_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    # 初始化列表来存储质心数据
    centroids = []
    # 按每五行分组
    for i in range(0, len(df), 5):
        group = df.iloc[i:i+5]
        # 提取所有x和y坐标
        x_coords = group[['x1', 'x2']].values.flatten()
        y_coords = group[['y1', 'y2']].values.flatten()
        # 计算平均值
        centroid_x = x_coords.mean()
        centroid_y = y_coords.mean()
        centroids.append({'centroid_x': centroid_x, 'centroid_y': centroid_y})
    # 创建一个新的DataFrame
    centroids_df = pd.DataFrame(centroids)
    # 保存到新的Excel文件
    centroids_df.to_excel(output_path, index=False)

# 确保输出目录存在
output_dir = r'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\flexibility'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 处理1到38场比赛的H和O队文件
for match in range(1, 39):
    for team in ['H', 'O']:
        input_file = rf'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\_eachmatch_\{match}_{team}_edge.xlsx'
        output_file = os.path.join(output_dir, f'{match}{team}_flex.xlsx')
        try:
            process_file(input_file, output_file)
            print(f'Processed {input_file} and saved to {output_file}')
        except FileNotFoundError:
            print(f'File not found: {input_file}')
        except Exception as e:
            print(f'An error occurred processing {input_file}: {e}')