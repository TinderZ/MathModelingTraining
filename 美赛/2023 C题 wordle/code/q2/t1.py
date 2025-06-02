# import pandas as pd
# from collections import defaultdict

# # 读取Excel文件
# df = pd.read_excel('new_wordattribute.xlsx', header=None, skiprows=1)

# # 假设Word在第3列，即索引为2
# i = 0
# words = df.iloc[:, 2].tolist()

# for word in words :
#     if not word.isalpha() or len(word) != 5 :
#         i+=1
#         print(word)
# print(i)

# filtered_words = [word.lower() for word in words]

# # 初始化字典来存储字母频率
# letter_freq = {chr(i): [0]*5 for i in range(97,123)}  # a-z for each position
# letter_tim = {chr(i): [0] for i in range(97,123)}

# # 统计每个字母在每个位置出现的次数
# for word in filtered_words:
#     for i, letter in enumerate(word):
#         letter_freq[letter][i] += 1
#         letter_tim[letter][0] += 1

# # 计算频率
# for letter in letter_freq:
#     for pos in range(5):
#         letter_freq[letter][pos] /= letter_tim[letter][0]

# #-------------------------------------------------------------------------------------------------------

# # 创建DataFrame并保存到Excel
# letters = [chr(i) for i in range(97,123)]
# freq_df = pd.DataFrame(letter_freq).T
# freq_df.columns = ['Position1', 'Position2', 'Position3', 'Position4', 'Position5']
# freq_df = freq_df.fillna(0)  # 填充缺失值为0
# freq_df.to_excel('q2\\letter_freq.xlsx')

# #格式化打印字母频率
# for letter in sorted(freq_df.index):
#     print(f"{letter}: {freq_df.loc[letter, 'Position1']:.3f}, {freq_df.loc[letter, 'Position2']:.3f}, {freq_df.loc[letter, 'Position3']:.3f}, {freq_df.loc[letter, 'Position4']:.3f}, {freq_df.loc[letter, 'Position5']:.3f}")

# #计算每个单词中字母的平均频率
# avg_freq = []
# for word in filtered_words:
#     frequencies = [letter_freq[letter][pos] for pos, letter in enumerate(word)]
#     average = sum(frequencies) / 5
#     avg_freq.append(average)

# # 将平均频率添加到DataFrame中
# df['Average Frequency'] = avg_freq
# df.to_excel('q2\\word_attribute_processed.xlsx', index=False)




import pandas as pd

# 读取字母频率数据
freq_df = pd.read_excel('q2\\letter_freq.xlsx', index_col=0)

# 读取原始单词数据
df_words = pd.read_excel('new_wordattribute.xlsx', header=None, skiprows=1)

# 提取并处理单词
words = df_words.iloc[:, 2].tolist()
filtered_words = [str(word).lower() for word in words if str(word).isalpha() and len(str(word)) == 5]

# 创建一个新的DataFrame来存储频率
word_freq = pd.DataFrame(columns=['Word', 'Position1', 'Position2', 'Position3', 'Position4', 'Position5'])

# 遍历每个单词
for word in filtered_words:
    freqs = []
    for pos in range(5):
        letter = word[pos]
        freq = freq_df.loc[letter, f'Position{pos+1}']
        freqs.append(freq)
    # 将数据添加到DataFrame中
    word_freq = word_freq._append({'Word': word,
                                  'Position1': freqs[0],
                                  'Position2': freqs[1],
                                  'Position3': freqs[2],
                                  'Position4': freqs[3],
                                  'Position5': freqs[4]}, ignore_index=True)

# 保存到Excel文件
word_freq.to_excel('q2\\word_attribute_processed3.xlsx', index=False)