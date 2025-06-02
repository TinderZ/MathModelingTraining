import osmnx as ox
import pandas as pd
import folium
import branca.colormap as cm

# 获取巴尔的摩的交通路线图
G = ox.graph.graph_from_place("Baltimore, Maryland, USA", network_type="drive")

# 加载公交站数据
bus_stops = pd.read_csv("2025_Problem_D_Data\\Bus_Stops.csv")

# 创建Folium地图
gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
center_lat = gdf_nodes['y'].mean()
center_lon = gdf_nodes['x'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# 添加道路网络到地图
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
folium.GeoJson(
    edges[['geometry']],
    style_function=lambda x: {
        'color': '#999999',
        'weight': 0.5,
        'opacity': 0.7
    }
).add_to(m)

# 创建颜色映射
min_rider = bus_stops['Stop_Rider'].min()
max_rider = bus_stops['Stop_Rider'].max()
colormap = cm.LinearColormap(
    colors=['#440154', '#21918c', '#fde725'],  # viridis颜色渐变
    vmin=min_rider,
    vmax=max_rider
)
colormap.caption = 'Bus Stop Ridership'
colormap.add_to(m)

# 添加公交站点到地图
for _, stop in bus_stops.iterrows():
    rider = stop['Stop_Rider']
    size = 1 + 5 * (rider - min_rider) / (max_rider - min_rider)  # 半径范围5-20像素
    
    folium.CircleMarker(
        location=[stop['Y'], stop['X']],  # 注意坐标顺序：纬度在前
        radius=size,
        color=colormap(rider),
        fill=True,
        fill_color=colormap(rider),
        fill_opacity=0.7,
        weight=0.5
    ).add_to(m)

# 保存地图
m.save('baltimore_bus_map.html')