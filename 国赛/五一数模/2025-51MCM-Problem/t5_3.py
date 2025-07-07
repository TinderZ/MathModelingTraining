import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取数据并生成特征
def generate_features(df):
    df = df.copy()
    if 'I' in df.columns:
        df.drop(columns=['I', 'x_squared', 'x_3', 'I_x', 'I_intercept'], inplace=True, errors='ignore')
    df['I'] = df['x'].apply(lambda x: 1 if 13 <= x <= 17 else 0)
    df['x_squared'] = df['x']**2
    df['x_3'] = df['x']**3
    df['I_x'] = df['I'] * df['x']
    df['I_intercept'] = df['I']
    return df[['x', 'x_squared', 'x_3', 'I_intercept', 'I_x']]

# 加载原始数据
df_all = pd.read_excel('待拟合函数.xlsx', sheet_name='Sheet1')
X_all = generate_features(df_all.copy())
y_all = df_all['y']

# 必须保留的点（x=8和x=24）
must_keep_indices = df_all[(df_all['x'] == 8) | (df_all['x'] == 24)].index.tolist()
current_df = df_all.copy()
removed_order = []
r2_history = []

# 初始模型（使用所有点）
model = LinearRegression()
model.fit(generate_features(current_df), current_df['y'])
initial_r2 = r2_score(y_all, model.predict(X_all))
r2_history.append(initial_r2)
print(f"初始R²: {initial_r2:.4f}")

# 逐步删除点
while True:
    best_r2 = -np.inf
    best_point = None
    candidates = [idx for idx in current_df.index if idx not in must_keep_indices]
    
    if not candidates:
        print("无法继续删除，所有非必须点已移除。")
        break
    
    # 遍历每个候选点，找出删除后R²最大的情况
    for candidate in candidates:
        temp_df = current_df.drop(candidate)
        X_temp = generate_features(temp_df)
        y_temp = temp_df['y']
        
        try:
            model = LinearRegression().fit(X_temp, y_temp)
            pred_all = model.predict(X_all)
            r2 = r2_score(y_all, pred_all)
        except:
            r2 = -np.inf  # 处理无法拟合的情况
        
        if r2 > best_r2:
            best_r2 = r2
            best_point = candidate
    
    # 记录结果并删除点
    if best_r2 < 0.9:
        print("R² < 0.9，停止删除。")
        break
    
    removed_order.append(current_df.loc[best_point, 'x'])
    current_df = current_df.drop(best_point)
    r2_history.append(best_r2)
    print(f"删除x={removed_order[-1]}, 当前R²: {best_r2:.4f}, 剩余点数: {len(current_df)}")

# 输出结果
print("\n删除顺序及R²变化：")
for x, r2 in zip(removed_order, r2_history[1:]):
    print(f"删除点 x={x:<3} → R²={r2:.4f}")

minimal_points = current_df['x'].tolist()
print(f"\n最少需要 {len(minimal_points)} 个点：{sorted(minimal_points)}")
print(f"最终R²: {r2_history[-1]:.4f}")