import csv

# 读取附件4中的数据
data = []
with open('附件4.csv', 'r', encoding='gbk') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        category1, category2, value = row
        data.append((category1, category2, int(value)))

# 构建COO格式的稀疏矩阵
values = []
row_indices = []
col_indices = []

category_to_index = {}
index_to_category = []


def get_index(category):
    if category not in category_to_index:
        index = len(index_to_category)
        category_to_index[category] = index
        index_to_category.append(category)
    return category_to_index[category]


for category1, category2, value in data:
    row = get_index(category1)
    col = get_index(category2)
    values.append(value)
    row_indices.append(row + 1)  # LINGO索引从1开始
    col_indices.append(col + 1)  # LINGO索引从1开始

# 输出结果
print("values =", values, ";")
print("row_indices =", row_indices, ";")
print("col_indices =", col_indices, ";")
