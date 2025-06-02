import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
import matplotlib.pyplot as plt
from collections import deque
import holidays

#1. 数据预处理
df = pd.read_excel('feature_extracted_data.xlsx', sheet_name='Sheet1', parse_dates=['Date'])
df.set_index('Date', inplace=True)
# 选择特征
features = ['Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7', 'IsHoliday', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7']
#features = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7']
X = df[features].values
y = df['Number of reported results'].values

# 归一化
scalerx = MinMaxScaler(feature_range=(0, 1))
scalery = MinMaxScaler(feature_range=(0, 1))
X_scaled = scalerx.fit_transform(X)
y_scaled = scalery.fit_transform(y.reshape(-1, 1))

def create_dataset(X, y, look_back=1):   #2. 创建时间序列数据集
    Xs, ys = [], []
    for i in range(len(X)-look_back):
        v = X[i:(i+look_back)]
        Xs.append(v)
        ys.append(y[i+look_back])
    return np.array(Xs), np.array(ys)


look_back = 14  # 可以根据需要调整


X_seq, y_seq = create_dataset(X_scaled, y_scaled, look_back)

#3. 划分训练集和验证集
train_size = int(len(X_seq) * 0.9)
X_train, X_val = X_seq[:train_size], X_seq[train_size:]
y_train, y_val = y_seq[:train_size], y_seq[train_size:]

# 4. 构建LSTM模型
model = Sequential()
model.add(LSTM(100, activation='tanh', input_shape=(look_back, len(features))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')


#5. 训练模型
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val))  

t = 60  #       ------------------------------------------------------------------------------------

# 6. 预测
# 滚动预测未来的值
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=t, freq='D')

# 计算未来日期的星期几
known_date = pd.Timestamp('2022-01-07')
known_weekday = 5  # 1表示周一，5表示周五

future_df = pd.DataFrame(index=future_dates)
future_df['day of the week'] = future_df.index.map(lambda date: ((known_weekday - 1) + (date - known_date).days) % 7 + 1)

for i in range(1, 8):
    future_df[f'Day_{i}'] = future_df['day of the week'].apply(lambda x: 1 if x == i else 0)

# 判断是否是节假日
us_holidays = holidays.US()
future_df['IsHoliday'] = future_df.index.map(lambda x: 1 if x in us_holidays else 0)

# # 初始化lag_queue，包含最后7个实际值
# lag_queue = deque(y_seq[-7:].flatten().tolist(), maxlen=7)

# 初始化queue2，包含最后7个单时间步的数据
last_7_steps = X_scaled[-look_back:]
queue2 = deque(last_7_steps, maxlen=look_back)

# 准备列表来保存未来t天的预测值
future_predictions_scaled = []

for i in range(t):
    # 获取最近7个时间步的数据
    input_sequence = np.array(list(queue2)).reshape(1, look_back, len(features))
    
    # 预测下一个值
    y_pred_scaled = model.predict(input_sequence)
    future_predictions_scaled.append(y_pred_scaled[0][0])
    
    # 获取未来第i天的星期特征和是否节假日特征
    day_holiday_features = future_df.loc[future_dates[i], ['Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7', 'IsHoliday']].values
    
    # 提取队列中最后一个时间步的lag特征
    last_lag_features = queue2[-1][8:15]  # Lag_1到Lag_7
    
    # 更新lag特征：Lag_2到Lag_7成为Lag_1到Lag_6，Lag_1是预测值
    new_lag_features = np.concatenate(([y_pred_scaled[0][0]], last_lag_features[:-1]))
    
    # 组合新的时间步特征
    new_time_step = np.concatenate((day_holiday_features, new_lag_features))
    
    # 将新的时间步加入queue2
    queue2.append(new_time_step)


# 7.反归一化
future_predictions = scalery.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

print("最后一天的预测值:", future_predictions[-1][0])





# 准备数据
future_results = pd.DataFrame({
    'Date': future_dates,
    'Predicted Results': future_predictions.flatten()
})

# 读取现有数据
file_path = 'result.xlsx'
sheet_name = 'Sheet1'
existing_data = pd.read_excel(file_path, sheet_name=sheet_name)

# 检查并追加数据
if existing_data.empty:
    # 如果Sheet1为空，直接写入新的预测数据
    future_results.to_excel(file_path, sheet_name=sheet_name, index=False)
else:
    # 如果Sheet1不为空，追加新的预测数据
    # 确保没有重复的列名
    if 'Date' in existing_data.columns and 'Predicted Results' in existing_data.columns:
        # 追加数据
        combined_data = pd.concat([existing_data, future_results], ignore_index=True)
    else:
        # 如果列名不匹配，可能需要调整列名或处理方式
        combined_data = existing_data.copy()
        # 这里可以根据实际情况进行处理
    # 写入数据
    combined_data.to_excel(file_path, sheet_name=sheet_name, index=False)







# 8.绘制实际值和预测值
plt.plot(df.index, df['Number of reported results'], label='Actual')
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=t)
plt.plot(future_dates, future_predictions, label='Predicted')
plt.legend()
plt.show()
