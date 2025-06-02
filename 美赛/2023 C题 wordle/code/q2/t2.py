import pandas as pd
from collections import defaultdict
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet

# 读取Excel文件
df = pd.read_excel('new_wordattribute.xlsx', header=None, skiprows=1)

# 假设Word在第3列，即索引为2
words = df.iloc[:, 2].tolist()

#判断每个单词的词性
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

pos_tags = pos_tag(words)
word_pos = []

for tag in pos_tags:
    wn_pos = get_wordnet_pos(tag[1])
    if wn_pos:
        if wn_pos == wordnet.NOUN:
            word_pos.append('n.')
        elif wn_pos == wordnet.VERB:
            word_pos.append('v.')
        elif wn_pos == wordnet.ADJ:
            word_pos.append('adj.')
        elif wn_pos == wordnet.ADV:
            word_pos.append('adv.')
        else:
            word_pos.append('other')
    else:
        word_pos.append('other')

df['Pos'] = word_pos

print(word_pos)


# 统计每个单词中出现最多的字母次数
def max_letter_count(word):
    count = {}
    for letter in word:
        count[letter] = count.get(letter, 0) + 1
    return max(count.values())

max_counts = [max_letter_count(word) for word in words]
df['Max Letter Count'] = max_counts

# 保存结果到新的Excel文件
df.to_excel('q2\\word_attribute_processed2.xlsx', index=False)






