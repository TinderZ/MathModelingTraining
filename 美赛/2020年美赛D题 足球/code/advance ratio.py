import pandas as pd
import os

# 创建一个空的DataFrame来存储结果
columns = ['Advance Ratio', 'Mean x1', 'Mean y1', 'Mean x2', 'Mean y2']
results = pd.DataFrame(columns=columns, index=range(1, 20))

for x in range(1, 20):
    # 构建文件路径
    filename = f"E:\\数模\\训练题\\2020年美赛D题 足球\\data\\passingevents\\Opponent{x}.xlsx"
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在，跳过。")
        continue
    
    # 读取Excel文件，跳过第一行标题
    df = pd.read_excel(filename, sheet_name='Sheet1', header=0)
    
    # 计算 advance ratio
    sum_N = df['abs delta Y'].sum()
    sum_abs_L = df['delta X'].abs().sum()
    advance_ratio = sum_N / sum_abs_L
    
    # 计算 H, I, J, K 列的均值
    mean_H = df['x1'].mean()
    mean_I = df['y1'].mean()
    mean_J = df['x2'].mean()
    mean_K = df['y2'].mean()
    
    # 将结果存储到DataFrame中
    results.loc[x] = [advance_ratio, mean_H, mean_I, mean_J, mean_K]

# 指定输出文件路径
output_path = "E:\\数模\\训练题\\2020年美赛D题 足球\\data\\passingevents\\advance ratio.xlsx"

# 将结果写入Excel文件
results.to_excel(output_path, index_label='Opponent')