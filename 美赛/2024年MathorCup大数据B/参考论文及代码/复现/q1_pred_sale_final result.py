# 随机挑选三个品类用于可视化
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取特征表格数据
feature_file_path = "销量预测特征表.csv"  # 请确保文件路径正确
df_features = pd.read_csv(feature_file_path)
all_categories = df_features['品类'].unique()

selected_categories = random.sample(list(all_categories), min(3, len(all_categories)))


prediction_results_df = pd.read_csv("品类销量预测结果.csv", encoding='gbk')

# 可视化随机挑选的品类的预测结果
for category in selected_categories:
    # 提取当前品类的预测结果
    df_category_predictions = prediction_results_df[prediction_results_df['品类'] == category]
    future_dates = df_category_predictions['日期']
    category_predictions = df_category_predictions['预测销量']

    # 绘制预测结果
    plt.figure(figsize=(10, 5))
    plt.plot(future_dates, category_predictions, marker='o', linestyle='-', color='b', label='预测销量')

    # color自己改颜色
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.title(f'{category} 品类的销量预测 (2023年7月-9月)')
    plt.xlabel('日期')
    plt.ylabel('预测销量')
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))  # 每隔10天显示一个标签
    plt.xticks(np.arange(1, 92, 10), rotation=45)  # 每隔10天显示一个标签，并旋转45度
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
