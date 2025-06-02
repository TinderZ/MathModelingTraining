import pandas as pd

# 读取Excel文件，跳过第一行，没有标题行
df = pd.read_excel('new_wordattribute.xlsx', header=None, skiprows=1)

# 提取第三列作为单词列表，并赋予列名'word'
words = df.iloc[:, 2]
#words = words.reset_index(drop=True)
words.name = 'word'

# 定义字母组合列表
bigrams = ['er', 'in', 'st', 'lo', 'al', 'ar', 'ch', 'or', 'th', 'an', 'ea', 'ro']

# 创建一个新的DataFrame来存储结果
result_df = words.to_frame()

for bigram in bigrams:
    # 检查单词是否包含字母组合，并将结果转换为整数（1或0）
    result_df[bigram] = words.str.contains(bigram, case=False, na=False).astype(int)

# 将结果保存到'zmzh.xlsx'文件中
result_df.to_excel('zmzh.xlsx', index=False)