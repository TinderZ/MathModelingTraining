import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
# 读取特征表格数据
feature_file_path = "销量预测特征表.csv"  # 请确保文件路径正确
df_features = pd.read_csv(feature_file_path)

# 获取所有品类
all_categories = df_features['品类'].unique()

# 初始化字典来存储每个品类的评估结果
results = []

# 遍历每个品类，训练模型并进行评估
for category in all_categories:
    # 提取当前品类的数据
    df_category = df_features[df_features['品类'] == category]

    # 检查样本数量是否足够进行训练和测试划分
    if len(df_category) < 5:
        print(f"品类 {category} 样本数量不足，跳过训练和评估。")
        continue

    # 准备训练数据和标签
    feature_columns = [f'星期_{i}' for i in range(1, 8)] + ['是否节假日', '是否促销日'] + [f'滞后_{lag}天' for lag in range(1, 8)]
    X = df_category[feature_columns]
    y = df_category['销量']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化随机森林模型
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)

    # 训练模型
    rf_model.fit(X_train, y_train)

    # 使用测试集进行预测
    y_pred = rf_model.predict(X_test)

    # 过滤掉真实值为 0 的数据点，避免 MAPE 计算时除以 0
    non_zero_indices = y_test != 0
    y_test_filtered = y_test[non_zero_indices]
    y_pred_filtered = y_pred[non_zero_indices]

    # 检查是否有有效的数据点进行评估
    if len(y_test_filtered) == 0:
        print(f"品类 {category} 在测试集中所有真实销量为 0，无法计算 MAPE，跳过评估。")
        continue

    # 评估模型表现 (MAPE)
    mape = np.mean(np.abs((y_test_filtered - y_pred_filtered) / y_test_filtered)) * 100

    # 保存评估结果
    results.append({'品类': category, 'MAPE': mape})

    # 保存模型到文件
    model_file_path = f"trained_models/model_{category}.pkl"
    dump(rf_model, model_file_path)


# 转换结果为 DataFrame
results_df = pd.DataFrame(results)


print(results_df.head())

# 保存评估结果到文件
results_file_path = "品类销量预测评估结果.csv"
results_df.to_csv(results_file_path, index=False)

print(f"模型评估完成，MAPE结果已保存至 {results_file_path}")
