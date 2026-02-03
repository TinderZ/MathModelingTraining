import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免中文路径导致的Tcl问题
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import Rectangle

# =========================================================================
# Title: Power Consumption Analysis (Python Version)
# Description:
#   MCM Style Visualization Script
#   1. Process: Read CSV -> Smooth (10-min mean) -> Plot Stacked Area
#   2. Feature: Main Plot + Zoom Inset (Raw Data) + Full Legend with Frame
# =========================================================================

# --- 0. 参数设置 (Parameters) ---
FILE_PATH = 'data.csv'
GROUP_SIZE = 3  # 每10个点取一次均值
ZOOM_START = 100  # 局部放大起始行索引
ZOOM_END = 150  # 局部放大结束行索引

START_TIME = '2024-07-12 06:00:00'
END_TIME = '2024-07-12 13:00:00'

# 设置字体为 Times New Roman (美赛标准)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 配色设置 - 方案1: 原配色 (ColorBrewer风格)
COLORS_ORIGINAL = [
    '#4575b4',  # Blue
    '#fc8d59',  # Orange
    '#91cf60',  # Green
    '#c2a5cf',  # Purple
    '#d8daeb',  # Grey-ish
    '#99d594'   # Light Green
]

# 配色设置 - 方案2: 莫兰迪色系 (Morandi Colors，低饱和度高级感)
COLORS_MORANDI = [
    '#A8B8A0', # 莫兰迪蓝 (雾霾蓝)
    '#8EAFC4',  # 莫兰迪绿 (灰豆绿)
    '#D4A59A',  # 莫兰迪粉橘
    '#B8A9C9',  # 莫兰迪紫
    '#E8D5B7',  # 莫兰迪米黄
    '#C9887A'   # 莫兰迪青
]

# 选择使用哪组配色 (切换这里即可)
COLORS_STACK = COLORS_MORANDI  # 当前使用莫兰迪色系
COLOR_REAL = '#7A8B8B'  # 莫兰迪灰 (Real Power)

# 自定义分功耗名称 (下标格式)
LEGEND_LABELS = [r'$P_{cpu}$', r'$P_{screen}$', r'$P_{cellular}$', r'$P_{wifi}$', r'$P_{gps}$', r'$P_{background}$']

# --- 1. 数据读取与处理 (Data Processing) ---
try:
    # 读取数据 (假设无表头，或者第一行是表头但我们只关心列的位置)
    # header=0 表示第一行是标题，如果您的csv没有标题，请改用 header=None
    df = pd.read_csv(FILE_PATH)

    # 确保至少有2列
    if df.shape[1] < 2:
        raise ValueError("Data must have at least 2 columns.")

    # 提取数据（跳过第一列timestamp）
    raw_real = df.iloc[:, 1].values  # 第二列：真实总功耗 (P_real)
    raw_comps = df.iloc[:, 2:].values  # 第三列及以后：分功耗
    num_comps = raw_comps.shape[1]

    # 确保颜色足够
    if num_comps > len(COLORS_STACK):
        import matplotlib.cm as cm

        COLORS_STACK = [cm.tab10(i) for i in range(num_comps)]

    # 生成时间轴
    time_index = pd.date_range(start=START_TIME, end=END_TIME, periods=len(df))
    df_combined = pd.DataFrame(raw_comps, index=time_index)
    df_combined['Real_Power'] = raw_real

except Exception as e:
    print(f"Error reading data: {e}")
    # 生成模拟数据以供演示 (防止运行时无文件报错)
    print("Generating mock data for demonstration...")
    periods = 2000
    time_index = pd.date_range(start=START_TIME, end=END_TIME, periods=periods)
    raw_comps = np.random.rand(periods, 4) * 10 + 5  # 4个分量
    raw_real = np.sum(raw_comps, axis=1) + np.random.normal(0, 2, periods)  # 真实值 = 总和 + 噪声
    num_comps = 4
    df_combined = pd.DataFrame(raw_comps, index=time_index)
    df_combined['Real_Power'] = raw_real

# --- 2. 数据平滑 (Smoothing) ---
# 使用 groupby 对每 N 行进行聚合平均
# 这种方法比 resample('10T') 更精确对应 MATLAB 的 "每10个点一组"
smooth_df = df_combined.groupby(np.arange(len(df_combined)) // GROUP_SIZE).mean()

# 恢复平滑后的时间轴 (取每组的第一个时间点或平均时间点)
smooth_time = time_index[::GROUP_SIZE][:len(smooth_df)]

# 分离平滑数据
smooth_comps = smooth_df.iloc[:, :num_comps].values
smooth_real = smooth_df['Real_Power'].values

# --- 3. 绘图 (Plotting) ---
fig, ax = plt.subplots(figsize=(12, 7))

# === 主图 (Main Plot) ===

# 3.1 绘制堆积图 (Simulated Components)
current_labels = LEGEND_LABELS[:num_comps] if num_comps <= len(LEGEND_LABELS) else [f'Comp {i}' for i in
                                                                                    range(num_comps)]
stacks = ax.stackplot(smooth_time, smooth_comps.T, labels=current_labels,
                      colors=COLORS_STACK[:num_comps], alpha=1, zorder=1)

# 3.2 绘制真实值曲线 (Real Power) - 虚线，放在最前面
line_real, = ax.plot(smooth_time, smooth_real, color=COLOR_REAL, linewidth=2, 
                     linestyle='--', label='Real Total Power', zorder=3)

# 3.3 设置坐标轴格式
ax.set_xlim(smooth_time[0], smooth_time[-1])  # 使用平滑后的时间范围，消除右侧空白
# Y轴范围：自动根据数据计算，留10%余量
y_max = max(smooth_real.max(), smooth_comps.sum(axis=1).max()) * 1.1
ax.set_ylim(0, y_max)
ax.set_xlabel('Time (MM-DD HH)', fontsize=14, fontname='Times New Roman')
ax.set_ylabel('Power Consumption (mW)', fontsize=14, fontname='Times New Roman')  # 单位改为mW

# 格式化 X 轴时间显示 (例如 07-12 06)
date_fmt = mdates.DateFormatter('%m-%d %H')
ax.xaxis.set_major_formatter(date_fmt)
plt.xticks(rotation=0)

# 3.4 图例设置 (Legend with Solid Box) - 放在右上角
handles = [line_real] + stacks
labels_all = ['Real Total Power'] + current_labels
# loc='upper right'
ax.legend(handles=handles, labels=labels_all, loc='upper right',
          frameon=True, framealpha=1, edgecolor='black', fancybox=False, fontsize=10)

# === 插图 (Inset Plot): 局部放大 === (已注释)
# # 调整位置：向左向上移动
# # [Left, Bottom, Width, Height]
# # 原 [0.22, 0.55, 0.35, 0.35] -> [0.15, 0.60, 0.35, 0.35]
# axins = ax.inset_axes([0.15, 0.60, 0.35, 0.35])
#
# # 准备原始数据片段
# subset_time = time_index[ZOOM_START:ZOOM_END]
# subset_comps = df_combined.iloc[ZOOM_START:ZOOM_END, :num_comps].values
# subset_real = df_combined.iloc[ZOOM_START:ZOOM_END, num_comps].values  # Real Power col index
#
# # 4.1 绘制原始堆积图
# axins.stackplot(subset_time, subset_comps.T, colors=COLORS_STACK[:num_comps], alpha=0.95)
#
# # 4.2 绘制原始真实曲线
# axins.plot(subset_time, subset_real, color=COLOR_REAL, linewidth=1.5)
#
# # 4.3 插图设置
# # 不显示标题、不显示X/Y轴刻度标签(Labels)，但保留Tick marks以示存在
# axins.set_xticklabels([])
# axins.set_yticklabels([])
# axins.set_xlim(subset_time[0], subset_time[-1])
#
# # 设置主图框的高度为3500 (通过设置子图的Y轴范围来实现)
# axins.set_ylim(0, 3500)
#
# axins.tick_params(axis='both', which='both', length=2)  # 刻度线缩短一点
# axins.grid(True, linestyle='--', alpha=0.5)
#
# # === 视觉引导：矩形框 + 箭头 ===
# # 1. 在主图绘制矩形框 (代替 mark_inset，因为不需要连接线)
# x_start_num = mdates.date2num(subset_time[0])
# x_end_num = mdates.date2num(subset_time[-1])
# width_days = x_end_num - x_start_num
#
# # 绘制深灰色实线框
# rect = Rectangle((x_start_num, 0), width_days, 3500,
#                  linewidth=1.5, edgecolor='dimgray', facecolor='none', linestyle='-')
# ax.add_patch(rect)
#
# # 2. 绘制箭头 (从框上方指向子图)
# # 箭头起点 (Tail): 框的上方中心 (Data Coordinates)
# arrow_tail_x = x_start_num + width_days / 2
# arrow_tail_y = 3500
#
# # 箭头终点 (Head): 子图的下方中心 (Axes Fraction Coordinates)
# # 子图位置: [0.15, 0.60, 0.35, 0.35]
# arrow_head_x = 0.15 + 0.35 / 2  # = 0.325
# arrow_head_y = 0.60  # 子图底部
#
# ax.annotate("",
#             xy=(arrow_head_x, arrow_head_y), xycoords='axes fraction',
#             xytext=(arrow_tail_x, arrow_tail_y), textcoords='data',
#             arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.5))

plt.tight_layout()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('power_consumption.png', dpi=300, bbox_inches='tight')
print("Figure saved to power_consumption.png")