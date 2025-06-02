import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('q2\\wordattribute.xlsx', sheet_name='ratio', index_col=None)

# 对 'POS Tagging' 列进行独热编码
pos_dummies = pd.get_dummies(df['POS Tagging'], prefix='POS')

# 合并原始数据框与独热编码后的列
df_encoded = pd.concat([df.drop('POS Tagging', axis=1), pos_dummies], axis=1)

# 保存更新后的数据框到新的 Excel 文件
df_encoded.to_excel('output.xlsx', index=False)

print("独热编码完成，并保存到 output.xlsx")











