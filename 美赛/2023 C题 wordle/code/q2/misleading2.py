import pandas as pd
import wordfreq
import string

# 读取Excel文件
df = pd.read_excel('new_wordattribute2.xlsx', header=None, skiprows=1)
# 提取并过滤单词，确保是字符串且小写
words = [str(word).lower() for word in df.iloc[:, 2].tolist() if isinstance(word, str)]

lang = 'en'

# 定义计算第二种误导度的函数
def calculate_misleading_v2(word, threshold=2):
    misleading_count = 0
    alphabet = string.ascii_lowercase
    # 定义相邻位置对的索引
    adjacent_pairs = [(0,1), (1,2), (2,3), (3,4)]
    for pair in adjacent_pairs:
        # 确定固定的位置索引
        fixed_positions = set(range(5)) - set(pair)
        # 遍历所有可能的字母组合来替换这两个位置
        for letter1 in alphabet:
            for letter2 in alphabet:
                if letter1 == word[pair[0]] and letter2 == word[pair[1]]:
                    continue  # 原字母不计数
                # 生成新的候选单词
                candidate = list(word)
                candidate[pair[0]] = letter1
                candidate[pair[1]] = letter2
                candidate_word = ''.join(candidate)
                # 检查单词频率
                if wordfreq.zipf_frequency(candidate_word, lang) >= threshold:
                    misleading_count += 1
    return misleading_count

# 计算每个单词的第二种误导度
results_v2 = []
for word in words:
    misleading_v2 = calculate_misleading_v2(word, threshold=2)
    results_v2.append((word, misleading_v2))

# 保存结果到Excel文件
df_results_v2 = pd.DataFrame(results_v2, columns=['Word', 'Misleading Count V2'])
df_results_v2.to_excel('word_misleading_counts_v2.xlsx', index=False)