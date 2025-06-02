import pandas as pd
from collections import Counter

# # 读取Excel文件，无表头，跳过第一行
# df = pd.read_excel('new_wordattribute2.xlsx', header=None, skiprows=1)

# # 获取第三列的单词列表，索引为2
# words = df.iloc[:, 2].tolist()

# # 筛选出长度为5的单词
# words_length_5 = [word for word in words if len(word) == 5]

# # 定义元音字母集合
# vowels = set('aeiou')

# # 统计每个单词中不同的元音字母数量
# vowel_counts = []
# for word in words_length_5:
#     word_lower = word.lower()
#     word_vowels = set(word_lower) & vowels
#     vowel_counts.append(len(word_vowels))

# # 创建DataFrame for individual words
# df_words = pd.DataFrame({
#     '单词': words_length_5,
#     '不同元音数量': vowel_counts
# })

# # 统计不同元音数量的分布
# count_distribution = Counter(vowel_counts)
# df_distribution = pd.DataFrame.from_dict(count_distribution, orient='index', columns=['频率'])
# df_distribution.index.name = '不同元音数量'

# # 保存到同一个Excel文件的不同sheet里
# with pd.ExcelWriter('yynum.xlsx') as writer:
#     df_words.to_excel(writer, sheet_name='单词列表', index=False)
#     df_distribution.to_excel(writer, sheet_name='分布统计')


# 读取Excel文件，无表头，跳过第一行
df = pd.read_excel('new_wordattribute2.xlsx', header=None, skiprows=1)

# 获取第三列的单词列表，索引为2
words = df.iloc[:, 2].tolist()

# 筛选出长度为5的单词
words_length_5 = [word for word in words if len(word) == 5]

# 定义元音字母集合
vowels = set('aeiou')

# 判断每个位置是否是元音
results = []
for word in words_length_5:
    word_lower = word.lower()  # 转换为小写
    is_vowel = [char in vowels for char in word_lower]  # 判断每个字符是否是元音
    results.append(is_vowel)

# 创建DataFrame
df_results = pd.DataFrame(results, columns=['is_vowel1', 'is_vowel2', 'is_vowel3', 'is_vowel4', 'is_vowel5'])
df_results.insert(0, 'word', words_length_5)  # 在第一列插入单词

# 保存到Excel文件
df_results.to_excel('yynum_vowel_positions.xlsx', index=False)

