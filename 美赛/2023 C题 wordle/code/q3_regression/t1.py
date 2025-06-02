import pandas as pd
import numpy as np
from scipy.stats import norm, chisquare

# 读取Excel文件
df = pd.read_excel('new_wordattribute.xlsx', header=0)

# 定义尝试次数的列名
try_columns = [f'{i} tries' for i in range(1,7)] + ['7 or more tries (X)']


# 创建一个新的DataFrame来存储均值和方差
results = pd.DataFrame(columns=['Word', 'Mean', 'Variance'])


for index, row in df.iterrows():
    # 提取尝试次数的频率
    freq = row[try_columns]
    
    # 计算总尝试次数
    total = freq.sum()
    
    if total == 0:
        continue  # 跳过总尝试次数为0的行
    
    # 尝试次数的列表
    tries = np.arange(1,8)  # 1到7


    # 计算均值
    mu = np.sum(tries * freq) / total
    
    # 计算方差
    var = np.sum((tries - mu)**2 * freq) / total
    sigma = np.sqrt(var)
    
    # 将结果添加到新的DataFrame中
    results = results._append({
        'Word': row['Word'],  # 假设你的DataFrame中有一个'Word'列
        'Mean': mu,
        'Variance': var
    }, ignore_index=True)

# 保存到新的Excel文件
results.to_excel('mean_variance_results.xlsx', index=False)

