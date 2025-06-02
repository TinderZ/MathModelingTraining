import pandas as pd
import wordfreq
import string

# 读取Excel文件
df = pd.read_excel('new_wordattribute2.xlsx', header=None, skiprows=1)
# 提取并过滤单词，确保是字符串且小写
words = [str(word).lower() for word in df.iloc[:, 2].tolist() if isinstance(word, str)]
#print(words)

# 设置语言为英语
lang = 'en'
# # 加载英语单词频率字典
# en_freq = wordfreq.zipf_frequency('en')
# print(en_freq[words[2]])


# 定义计算误导度的函数
def calculate_misleading(word, threshold=2):
    misleading_count = 0
    alphabet = string.ascii_lowercase
    for i in range(5):
        original_letter = word[i]
        for letter in alphabet:
            if letter == original_letter:
                continue
            candidate = word[:i] + letter + word[i+1:]
            #print(en_freq[candidate])
            if wordfreq.zipf_frequency(candidate, lang) >= threshold:
                misleading_count += 1
    return misleading_count

# 计算每个单词的误导度
results = []
for word in words:
    misleading = calculate_misleading(word, threshold=2)
    results.append((word, misleading))

# 保存结果到Excel文件
df_results = pd.DataFrame(results, columns=['Word', 'Misleading Count'])
df_results.to_excel('word_misleading_counts.xlsx', index=False)