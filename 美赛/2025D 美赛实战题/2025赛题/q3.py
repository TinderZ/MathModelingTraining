import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# 获取巴尔的摩的交通路线图
G = ox.graph.graph_from_place("Baltimore, Maryland, USA", network_type="drive")

# 加载公交站数据
bus_stops = pd.read_csv("2025_Problem_D_Data\\Bus_Stops.csv")

# 创建一个新的图形
fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='white')

# 获取 Stop_Rider 的最大值和最小值
min_rider = bus_stops['Stop_Rider'].min()
max_rider = bus_stops['Stop_Rider'].max()

# 创建颜色映射
cmap = cm.get_cmap('viridis')

# 遍历公交站数据
for index, stop in bus_stops.iterrows():
    # 获取公交站的经纬度
    y, x = stop['Y'], stop['X']
    
    # 获取 Stop_Rider 值
    stop_rider = stop['Stop_Rider']
    
    # 设置点的大小和颜色
    size = 200 * (stop_rider - min_rider) / (max_rider - min_rider)  # 根据 Stop_Rider 调整大小
    color = cmap((stop_rider - min_rider) / (max_rider - min_rider))  # 根据 Stop_Rider 设置颜色
    
    # 在图上绘制点
    scatter = ax.scatter(x, y, c=color, s=size, alpha=0.7, edgecolor='white', linewidths=0.1)

# 添加颜色条
plt.colorbar(scatter, label='Color Value')

# 添加图例和标题
ax.legend()
ax.set_title("Bus Transport Network")
plt.show()