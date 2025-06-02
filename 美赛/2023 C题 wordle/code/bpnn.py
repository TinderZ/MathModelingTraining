import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers, models, regularizers, optimizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# 读取Excel文件
df = pd.read_excel('new_wordattribute (3)_版本1_min-max标准化.xlsx', sheet_name='编码数据_版本1_min-max标准化', header=0)

# # Clean column names by removing '<br>' tags
# df.columns = df.columns.str.replace('<br>', '')

# # Specify the features without '<br>' tags
# features = [
#     'wordFrequency',
#     'Max Letter Count',
#     'freq of Position1',
#     'freq of Position2',
#     'freq of Position3',
#     'freq of Position4',
#     'freq of Position5',
#     'letter Average Frequency',
#     'Misleading Count',
#     'Misleading Count V2',
#     '不同元音数量'
# ]

# targets = ['1 tries', '2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)']

# 分割训练集和测试集
train_size = int(0.8 * len(df))

X_train = df[df.columns[15:]].iloc[:train_size].values
X_test = df[df.columns[15:]].iloc[train_size:].values

Y_train = df[df.columns[8:15]].iloc[:train_size].values
Y_test = df[df.columns[8:15]].iloc[train_size:].values



# 构建BP神经网络模型
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001), input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.002)),
    keras.layers.Dense(7, activation='linear')  # 输出层，7个目标变量
])

# 使用Adam优化器，并设置学习率
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, Y_train, epochs=100, batch_size=8, validation_split=0.1)

# 测试模型
Y_pred = model.predict(X_test)

# 评估模型
mse_list = []
mae_list = []
r2_list = []

for i in range(Y_test.shape[1]):
    mse = mean_squared_error(Y_test[:, i], Y_pred[:, i])
    mae = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
    r2 = r2_score(Y_test[:, i], Y_pred[:, i])
    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)
    print(f'目标 {i}:')
    print(f'  均方误差 (MSE): {mse}')
    print(f'  平均绝对误差 (MAE): {mae}')
    print(f'  决定系数 R²: {r2}')

# 计算平均值
avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)
avg_r2 = np.mean(r2_list)
print(f'平均 MSE: {avg_mse}')
print(f'平均 MAE: {avg_mae}')
print(f'平均 R²: {avg_r2}')

# # 绘制学习曲线
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()