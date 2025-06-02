import pandas as pd

# 读取CSV文件
file_path = 'output_OOO9.csv'
df = pd.read_csv(file_path)

# 提取所有唯一的ID并按字母顺序排序
unique_ids = sorted(pd.concat([df['Source'], df['Target']]).unique())

# 创建ID到数字的映射字典
id_map = {id_: idx + 1 for idx, id_ in enumerate(unique_ids)}

# 替换Source和Target列中的ID
df['Source'] = df['Source'].map(id_map)
df['Target'] = df['Target'].map(id_map)

# 确保Source和Target列是整数类型
df['Source'] = df['Source'].astype(int)
df['Target'] = df['Target'].astype(int)

# 保存到新的CSV文件
df.to_csv('output_OOO9.csv', index=False)