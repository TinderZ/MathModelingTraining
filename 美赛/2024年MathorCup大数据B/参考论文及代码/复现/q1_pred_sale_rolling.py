import pandas as pd
import numpy as np
import joblib
import os

# 未来预测的日期范围
future_dates = pd.date_range(start="2023-07-01", end="2023-09-30")

# 获取所有品类
all_categories = [f.replace("model_", "").replace(".pkl", "") for f in os.listdir("trained_models") if
                  f.endswith(".pkl")]

# 初始化用于保存预测结果的列表
prediction_results = []

# 遍历所有品类进行预测
for category in all_categories:
    model_path = f"trained_models/model_{category}.pkl"
    rf_model = joblib.load(model_path)

    # 创建一个 DataFrame 用于存储未来日期的特征
    df_future = pd.DataFrame({'日期': future_dates})
    # df_future['月'] = df_future['日期'].dt.month
    # df_future['日'] = df_future['日期'].dt.day

    df_future['星期几'] = df_future['日期'].apply(
        lambda date: ((5 - 1) + (date - pd.Timestamp('2022-07-01')).days) % 7 + 1)


    for i in range(1, 8):
        df_future[f'星期_{i}'] = df_future['星期几'].apply(lambda x: 1 if x == i else 0)

    # 如果你不想在原始 DataFrame 中保留 '星期几' 列，可以选择删除它
    df_future = df_future.drop('星期几', axis=1)

    df_future['是否节假日'] = df_future['日期'].apply(lambda date: 1 if date in [
        pd.Timestamp('2023-04-05'),  # 清明节
        pd.Timestamp('2023-05-01'),  # 劳动节
        pd.Timestamp('2023-05-14'),  # 母亲节
        pd.Timestamp('2023-05-20'),
        pd.Timestamp('2023-05-21'),
        pd.Timestamp('2023-06-01'),  # 儿童节
        pd.Timestamp('2023-06-18'),  # 父亲节
        pd.Timestamp('2023-06-22'),  # 端午节
        pd.Timestamp('2022-08-04'),  # 七夕情人节
        pd.Timestamp('2023-08-22'),  # 七夕情人节
        pd.Timestamp('2022-09-10'),  # 中秋节
        pd.Timestamp('2023-09-29'),  # 中秋节
    ] else 0)
    df_future['是否促销日'] = df_future['日期'].apply(lambda date: 1 if date in (
        list(pd.date_range(start=pd.Timestamp('2023-06-01'), end=pd.Timestamp('2023-06-18'))) +
        [pd.Timestamp('2022-08-08'), pd.Timestamp('2022-08-18'),  # 8 8
         pd.Timestamp('2023-08-08'), pd.Timestamp('2023-08-18'),
         pd.Timestamp('2022-09-09'), pd.Timestamp('2023-09-09'),  # 99 大促
         ]) else 0)

    # 使用最后的已知历史数据作为起始滞后特征
    df_category = pd.read_csv("销量预测特征表.csv")
    df_category = df_category[df_category['品类'] == category]
    last_known_data = df_category.iloc[-7:]['销量'].values.tolist()

    for current_date in future_dates:
        # 构造当前日期的特征
        current_features = df_future[df_future['日期'] == current_date][
            [f'星期_' + str(i) for i in range(1, 8)] + ['是否节假日', '是否促销日']].copy()
        # 添加滞后特征
        for lag in range(1, 8):
            if len(last_known_data) >= lag:
                current_features[f'滞后_{lag}天'] = last_known_data[-lag]
            else:
                current_features[f'滞后_{lag}天'] = np.nan  # 如果数据不足，可以考虑填充均值等

        # 预测当前日期的销量
        current_prediction = rf_model.predict(current_features)[0]

        # 更新预测结果
        prediction_results.append({'品类': category, '日期': current_date, '预测销量': current_prediction})

        # 更新已知数据，用于下一个日期的滞后特征
        last_known_data.append(current_prediction)

        # 只保留最近 7 天的数据
        if len(last_known_data) > 7:
            last_known_data.pop(0)

# 转换预测结果为 DataFrame
prediction_results_df = pd.DataFrame(prediction_results)

# 保存预测结果到文件
prediction_results_file_path = "品类销量预测结果_逐步预测.csv"
prediction_results_df.to_csv(prediction_results_file_path, index=False)

print(f"未来销量预测完成，结果已保存至 {prediction_results_file_path}")
