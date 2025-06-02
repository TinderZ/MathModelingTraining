import pandas as pd
import wordfreq
import string

# 设置语言为英语
lang = 'en'
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

re = calculate_misleading_v2('eeire')
print(re)