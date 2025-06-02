# 导入所需库
import pandas as pd
import numpy as np
import random

# 文件路径
inventory_forecast_path = '所有品类库存量预测结果.csv'
sales_forecast_path = '最终日销量预测结果.csv'
warehouse_info_path = '附件3.csv'
category_correlation_path = '附件4.csv'

# 读取数据
inventory_forecast_df = pd.read_csv(inventory_forecast_path, encoding='gbk')
sales_forecast_df = pd.read_csv(sales_forecast_path)
warehouse_info_df = pd.read_csv(warehouse_info_path, encoding='gbk')
category_correlation_df = pd.read_csv(category_correlation_path, encoding='gbk')

# 数据预处理
inventory_forecast_df.rename(columns={'Unnamed: 0': '品类'}, inplace=True)

# 提取仓库容量、产能、租金成本
warehouse_capacity = warehouse_info_df.set_index('仓库')['仓容上限'].to_dict()
warehouse_output_capacity = warehouse_info_df.set_index('仓库')['产能上限'].to_dict()
warehouse_rent = warehouse_info_df.set_index('仓库')['仓租日成本'].to_dict()

# 提取库存量和销量预测
inventory_forecast = inventory_forecast_df.set_index('品类').to_dict('index')
sales_forecast = sales_forecast_df.set_index('品类').to_dict('index')


# 提取品类关联度
def extract_correlation(df):
    correlation_dict = {}
    for _, row in df.iterrows():
        correlation_dict[(row['品类1'], row['品类2'])] = row['关联度']
    return correlation_dict


category_correlation = extract_correlation(category_correlation_df)

# 定义模拟退火算法参数
initial_temperature = 10000
temperature_decay = 0.995
min_temperature = 1
iterations_per_temp = 100

# 初始化分配方案
categories = inventory_forecast_df['品类'].tolist()
warehouses = list(warehouse_capacity.keys())
current_solution = {category: random.choice(warehouses) for category in categories}


# 计算目标函数值
def calculate_cost(solution):
    cost_term = sum(warehouse_rent[solution[i]] for i in categories)
    utilization_term = 0
    correlation_term = 0

    # 计算仓容利用率（包括7月、8月、9月）
    for j in warehouses:
        total_inventory_7 = sum(inventory_forecast[i]['7月库存量'] for i in categories if solution[i] == j)
        total_inventory_8 = sum(inventory_forecast[i]['8月库存量'] for i in categories if solution[i] == j)
        total_inventory_9 = sum(inventory_forecast[i]['9月库存量'] for i in categories if solution[i] == j)
        total_output = sum(sales_forecast[i]['7月1日'] for i in categories if solution[i] == j)

        # 添加惩罚项，如果超过仓容或产能上限
        if total_inventory_7 > warehouse_capacity[j] or total_inventory_8 > warehouse_capacity[j] or total_inventory_9 > \
                warehouse_capacity[j]:
            utilization_term += 1e10  # 更大的惩罚项
        if total_output > warehouse_output_capacity[j]:
            utilization_term += 1e10  # 更大的惩罚项

    # 计算品类关联度
    for (i, k), correlation in category_correlation.items():
        if solution[i] == solution.get(k):
            correlation_term -= correlation

    return cost_term + utilization_term + correlation_term


# 初始化最优解
current_cost = calculate_cost(current_solution)
best_solution = current_solution.copy()
best_cost = current_cost

# 模拟退火过程
temp = initial_temperature
while temp > min_temperature:
    for _ in range(iterations_per_temp):
        # 生成一个新的邻域解
        new_solution = current_solution.copy()
        random_category = random.choice(categories)
        new_solution[random_category] = random.choice(warehouses)

        # 计算新解的成本
        new_cost = calculate_cost(new_solution)

        # 判断是否接受新解
        if new_cost < current_cost or random.uniform(0, 1) < np.exp((current_cost - new_cost) / temp):
            current_solution = new_solution
            current_cost = new_cost

            # 更新最优解
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost

    # 降低温度
    temp *= temperature_decay

# 保存分仓方案为 CSV 文件
allocation_df = pd.DataFrame(list(best_solution.items()), columns=['品类', '仓库'])
allocation_df.to_csv('一品一仓分仓方案_模拟退火_改进版.csv', index=False)
print("分仓方案已保存到：一品一仓分仓方案_模拟退火_改进版.csv")
