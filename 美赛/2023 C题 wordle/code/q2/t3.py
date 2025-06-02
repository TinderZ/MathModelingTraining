import pandas as pd

# 英语字母频率字典
letter_freq = {
    'a': 0.08167,
    'b': 0.01492,
    'c': 0.02782,
    'd': 0.04253,
    'e': 0.12702,
    'f': 0.02228,
    'g': 0.02015,
    'h': 0.06094,
    'i': 0.06966,
    'j': 0.00153,
    'k': 0.00772,
    'l': 0.04025,
    'm': 0.02406,
    'n': 0.06749,
    'o': 0.07507,
    'p': 0.01929,
    'q': 0.00095,
    'r': 0.05987,
    's': 0.06327,
    't': 0.09056,
    'u': 0.02758,
    'v': 0.00978,
    'w': 0.02360,
    'x': 0.00150,
    'y': 0.01974,
    'z': 0.00074
}

# 读取Excel文件
df = pd.read_excel('new_wordattribute2.xlsx', header=None, skiprows=1)

# 提取并过滤单词
words = df.iloc[:, 2].tolist()


# 计算平均频率
avg_freq = []
pos1_freq = []
pos2_freq = []
pos3_freq = []
pos4_freq = []
pos5_freq = []



for word in words:
    frequencies = [letter_freq[letter] for letter in word]
    average = sum(frequencies) / len(word)
    avg_freq.append(average)
    pos1_freq.append(frequencies[0])
    pos2_freq.append(frequencies[1])
    pos3_freq.append(frequencies[2])
    pos4_freq.append(frequencies[3])
    pos5_freq.append(frequencies[4])


# 创建DataFrame并保存到Excel
word_freq_df = pd.DataFrame({'Word': words, 'Average Frequency': avg_freq, 
                             'Position1': pos1_freq, 'Position2': pos2_freq, 
                             'Position3': pos3_freq, 'Position4': pos4_freq, 'Position5': pos5_freq})
word_freq_df.to_excel('q2\\word_attribute_processed5.xlsx', index=False)