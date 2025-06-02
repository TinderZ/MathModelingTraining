import pandas as pd
import wordfreq
import pandas as pd
from collections import Counter


# 读取Excel文件
df = pd.read_excel('new_wordattribute2.xlsx', header=None, skiprows=1)

# 提取并过滤单词
words = df.iloc[:, 2].tolist()
# 设置语言为英语
lang = 'en'

# 获取每个单词的频率
frequencies = [wordfreq.zipf_frequency(word, lang) for word in words]

print(wordfreq.zipf_frequency('eerie', 'en'))

df['Frequency'] = frequencies
#df.to_excel('q2\\word_attribute_processed66.xlsx', sheet_name='Sheet1', index=False)




# 计算所有相邻两字母搭配的频数
bigram_counter = Counter()

for word in words:
    if len(word) == 5:  # 确保单词长度为5
        for i in range(len(word) - 1):
            bigram = word[i:i+2]  # 获取相邻两字母搭配
            bigram_counter[bigram] += 1


print(bigram_counter)

# 计算总的两字母搭配次数
total_bigrams = sum(bigram_counter.values())

# 计算每个两字母搭配的频率
bigram_frequencies = {bigram: count / total_bigrams for bigram, count in bigram_counter.items()}

# 为每个5字母单词查找其4个搭配的频率
word_bigram_frequencies = {}

for word in words:
    if len(word) == 5:  # 确保单词长度为5
        bigrams = [word[i:i+2] for i in range(len(word) - 1)]  # 获取单词的所有相邻两字母搭配
        frequencies = [bigram_frequencies.get(bigram, 0) for bigram in bigrams]  # 查找每个搭配的频率
        word_bigram_frequencies[word] = frequencies

# 将结果保存到新的Excel文件
result_df = pd.DataFrame.from_dict(word_bigram_frequencies, orient='index', columns=['Bigram1', 'Bigram2', 'Bigram3', 'Bigram4'])
#result_df.to_excel('wa1.xlsx', index_label='Word')

print("结果已保存到 wa1.xlsx")











