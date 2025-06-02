import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GM_model import GM11, DGM21

# 读取库存量数据
inventory_file_path = "附件1.csv"  # 请确保文件路径正确
df_inventory = pd.read_csv(inventory_file_path, encoding='gbk')

# 转换月份列为 pandas 日期类型
df_inventory['月份'] = pd.to_datetime(df_inventory['月份'], format='%Y年%m月')

# 提取2023年1-6月的数据和2022年7-9月的数据
inventory_2023 = df_inventory[(df_inventory['月份'] >= '2023-01-01') & (df_inventory['月份'] <= '2023-06-01')]
inventory_2022 = df_inventory[(df_inventory['月份'] >= '2022-07-01') & (df_inventory['月份'] <= '2022-09-01')]

# 初始化字典来存储每个品类的预测结果
inventory_predictions = {}

# 获取所有品类
all_categories = df_inventory['品类'].unique()

# 对每个品类进行灰色预测 (GM(1,1))
for category in all_categories:
    # 获取该品类的今年和去年数据
    data = df_inventory[df_inventory['品类'] == category]['库存量'].values

    # 如果数据不足，跳过该品类
    if len(data) < 1:
        continue

    original_sequence = data


    # 使用GM(1,1)模型进行预测
    gm_model_result = GM11(original_sequence, 3)
    # gm_model = DGM21(original_sequence, 3)
    # 预测未来3个月的库存量（7月、8月、9月）
    # predicted_inventory = [gm_model(i) for i in range(len(original_sequence) + 1, len(original_sequence) + 4)]
    inventory_predictions[category] = gm_model_result

# 转换预测结果为 DataFrame
inventory_predictions_df = pd.DataFrame(inventory_predictions, index=['2022年7月', '2022年8月', '2022年9月',
                                                                      '2022年1月', '2022年2月', '2022年3月',
                                                                      '2022年4月', '2022年5月', '2022年6月',
                                                                      '2023年7月', '2023年8月', '2023年9月'])

# 保存预测结果到文件
inventory_predictions_file_path = "库存量预测结果.csv"
inventory_predictions_df.to_csv(inventory_predictions_file_path, index=True)

print(f"库存量预测完成，结果已保存至 {inventory_predictions_file_path}")


# 绘制品类11、121、251的历史值和预测值
categories_to_plot = ['category61', 'category121', 'category241']

# 创建一个新的图形
plt.figure(figsize=(12, 8))

# 定义颜色列表
colors = ['blue', 'green', 'red']

for idx, category in enumerate(categories_to_plot):
    # 获取该品类的历史数据
    historical_data_2023 = inventory_2023[inventory_2023['品类'] == category]['库存量'].values
    historical_data_2022 = inventory_2022[inventory_2022['品类'] == category]['库存量'].values

    # 获取预测数据
    predicted_data = inventory_predictions[category]

    # 创建时间轴
    time_axis_2022 = pd.date_range(start='2022-07-01', periods=len(historical_data_2022), freq='MS')
    time_axis_2023 = pd.date_range(start='2023-01-01', periods=len(historical_data_2023), freq='MS')
    time_axis_predicted1 = pd.date_range(start='2022-07-01', periods=len(historical_data_2022), freq='MS')
    time_axis_predicted2 = pd.date_range(start='2023-01-01', periods=len(historical_data_2023), freq='MS')
    time_axis_predicted3 = pd.date_range(start='2023-07-01', periods=len(historical_data_2022), freq='MS')

    # 获取当前品类的颜色
    color = colors[idx]

    # 绘制历史数据和预测数据
    plt.plot(time_axis_2022, historical_data_2022, 'o-', color=color, label=f'{category} 2022年历史值')
    plt.plot(time_axis_2023, historical_data_2023, 's-', color=color, label=f'{category} 2023年历史值')
    plt.plot(time_axis_predicted1, predicted_data[:len(time_axis_predicted1)], '^-', color=color, label=f'{category} 预测值')
    plt.plot(time_axis_predicted2, predicted_data[len(time_axis_predicted1):len(time_axis_predicted1)+len(time_axis_predicted2)], '^-', color=color)
    plt.plot(time_axis_predicted3, predicted_data[len(time_axis_predicted1)+len(time_axis_predicted2):], '^-', color=color)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
# 设置图形标题和标签
plt.title('品类 库存量历史值与预测值')
plt.xlabel('月份')
plt.ylabel('库存量')
plt.grid(True)
plt.legend()
# 显示图形
plt.show()
