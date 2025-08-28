#!/usr/bin/env python3
"""
搜索速度最大的点
在10-16s区间内以0.1s的步长搜索前3个节点中速度最大的点
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加父目录到Python路径以导入t4模块
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir / 't4'))

from position_velocity_calculator import PositionVelocityCalculator
from t4simulate import DragonRectangleSimulation

def find_max_velocity_in_interval():
    """
    在10-16s区间内搜索前3个节点中速度最大的点
    """
    print("="*60)
    print("搜索速度最大的点")
    print("时间区间: [10, 16]s")
    print("搜索步长: 0.1s")
    print("搜索范围: 前3个节点（龙头前把手、龙头后把手、后一节后把手）")
    print("="*60)
    
    # 创建仿真实例
    sim = DragonRectangleSimulation(pitch=170.0, debug=False)
    
    # 创建速度计算器
    calc = PositionVelocityCalculator(sim)
    
    # 搜索参数
    start_time = 14.0
    end_time = 15.0
    dt = 0.001
    
    # 生成时间点
    time_points = np.arange(start_time, end_time + dt, dt)
    
    # 存储结果
    results = []
    max_velocity = 0.0
    max_info = None
    
    print(f"开始搜索，总共{len(time_points)}个时间点...")
    
    for i, t in enumerate(time_points):
        # 更新仿真到当前时间
        sim.time = t
        sim.update_all_positions()
        
        # 计算当前时刻的速度
        velocities_cm_s = calc.calculate_velocities_at_time()
        velocities_m_s = velocities_cm_s / 100.0  # 转换为m/s
        
        # 检查前3个节点的速度
        for node_idx in range(min(3, len(velocities_m_s))):
            velocity = velocities_m_s[node_idx]
            
            # 节点名称
            if node_idx == 0:
                node_name = "龙头前把手"
            elif node_idx == 1:
                node_name = "龙头后把手"
            elif node_idx == 2:
                node_name = "后一节后把手"
            
            # 记录结果
            result = {
                'time': t,
                'node_index': node_idx,
                'node_name': node_name,
                'velocity_m_s': velocity,
                'position_x_m': sim.positions[node_idx][0] / 100.0,
                'position_y_m': sim.positions[node_idx][1] / 100.0,
                'phase': sim.node_phases[node_idx]
            }
            results.append(result)
            
            # 检查是否为最大速度
            if velocity > max_velocity:
                max_velocity = velocity
                max_info = result.copy()
        
        # 显示进度
        if (i + 1) % 10 == 0 or i == len(time_points) - 1:
            print(f"已处理 {i + 1}/{len(time_points)} 个时间点")
    
    print("\n" + "="*60)
    print("搜索完成！")
    print("="*60)
    
    # 输出最大速度信息
    if max_info:
        print(f"\n🎯 发现最大速度点:")
        print(f"   时间: {max_info['time']:.1f}s")
        print(f"   节点: {max_info['node_name']} (索引: {max_info['node_index']})")
        print(f"   速度: {max_info['velocity_m_s']:.6f} m/s")
        print(f"   位置: ({max_info['position_x_m']:.8f}, {max_info['position_y_m']:.8f}) m")
        print(f"   轨迹相位: {max_info['phase']}")
    
    # 创建DataFrame并保存结果
    df = pd.DataFrame(results)
    
    
    # 输出每个节点的最大速度统计
    print(f"\n📈 各节点最大速度统计:")
    for node_idx in range(3):
        node_data = df[df['node_index'] == node_idx]
        if not node_data.empty:
            max_vel = node_data['velocity_m_s'].max()
            max_time = node_data.loc[node_data['velocity_m_s'].idxmax(), 'time']
            node_name = node_data.iloc[0]['node_name']
            print(f"   {node_name}: {max_vel:.6f} m/s (t={max_time:.1f}s)")
    
    # 显示速度变化趋势
    print(f"\n📊 速度变化趋势分析:")
    for node_idx in range(3):
        node_data = df[df['node_index'] == node_idx].sort_values('time')
        if not node_data.empty:
            node_name = node_data.iloc[0]['node_name']
            velocities = node_data['velocity_m_s'].values
            avg_vel = np.mean(velocities)
            std_vel = np.std(velocities)
            print(f"   {node_name}:")
            print(f"     平均速度: {avg_vel:.6f} m/s")
            print(f"     速度标准差: {std_vel:.6f} m/s")
            print(f"     速度范围: [{np.min(velocities):.6f}, {np.max(velocities):.6f}] m/s")
    
    return df, max_info

def plot_velocity_curves(df, max_info, save_path=None):
    """
    绘制时间-速度曲线图
    
    Args:
        df: 包含所有结果的DataFrame
        max_info: 最大速度点信息
        save_path: 保存路径，如果为None则显示图形
    """
    print("\n📈 正在生成时间-速度曲线图...")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 定义颜色和线型
    colors = ['#464879', '#5D7DB3', '#D2C3D5']  
    markers = ['o', 's', '^']  # 圆形、方形、三角形
    linestyles = ['-', '--', '-.']  # 实线、虚线、点划线
    
    node_names = ['龙头前把手', '龙头后把手', '后一节后把手']
    
    # 为每个节点绘制曲线
    max_velocities = []
    for node_idx in range(3):
        node_data = df[df['node_index'] == node_idx].sort_values('time')
        if not node_data.empty:
            times = node_data['time'].values
            velocities = node_data['velocity_m_s'].values
            
            # 绘制主曲线
            ax.plot(times, velocities, 
                   color=colors[node_idx], 
                   linewidth=2.5, 
                   linestyle=linestyles[node_idx],
                   marker=markers[node_idx], 
                   markersize=4, 
                   markevery=5,  # 每5个点显示一个标记
                   label=node_names[node_idx],
                   alpha=0.8)
            
            # 找到该节点的最大速度点
            max_vel_idx = node_data['velocity_m_s'].idxmax()
            max_vel_point = node_data.loc[max_vel_idx]
            max_velocities.append(max_vel_point)
            
            # # 标注该节点的最大速度点
            # ax.scatter(max_vel_point['time'], max_vel_point['velocity_m_s'],
            #           color=colors[node_idx], s=100, marker='*', 
            #           edgecolors='black', linewidths=1.5, zorder=10)
            
            # # 添加文本标注
            # ax.annotate(f'{node_names[node_idx]}\n最大值\n{max_vel_point["velocity_m_s"]:.4f} m/s\nt={max_vel_point["time"]:.1f}s',
            #            xy=(max_vel_point['time'], max_vel_point['velocity_m_s']),
            #            xytext=(10, 15), textcoords='offset points',
            #            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[node_idx], alpha=0.7),
            #            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
            #                          color='black', lw=1),
            #            fontsize=9, ha='left')
    
    # 标注全局最大速度点
    if max_info:
        ax.scatter(max_info['time'], max_info['velocity_m_s'],
                  color='red', s=20, marker='o', 
                  edgecolors='black', linewidths=1, zorder=15)
        
        # 智能选择注释位置，根据最大速度点在时间轴上的位置
        time_center = (df['time'].min() + df['time'].max()) / 2
        
        if max_info['time'] < time_center:
            # 如果最大速度点在左半部分，注释放在右边
            xytext_offset = (80, -160)
            connection_style = 'arc3,rad=0.3'
        else:
            # 如果最大速度点在右半部分，注释放在左边
            xytext_offset = (-120, -160)
            connection_style = 'arc3,rad=-0.3'
        
        ax.annotate(f'全局最大速度\n{max_info["velocity_m_s"]:.6f} m/s\n{max_info["node_name"]}\nt={max_info["time"]:.3f}s',
                   xy=(max_info['time'], max_info['velocity_m_s']),
                   xytext=xytext_offset, textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red'),
                   arrowprops=dict(arrowstyle='->', connectionstyle=connection_style,
                                 color='red', lw=2),
                   fontsize=11, ha='left', weight='bold')
    
    # 设置坐标轴
    ax.set_xlabel('时间 (s)', fontsize=12, weight='bold')
    ax.set_ylabel('速度 (m/s)', fontsize=12, weight='bold')
    ax.set_title('板凳龙前3个节点速度变化曲线 (10-16s)', fontsize=14, weight='bold', pad=20)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置图例
    legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.9, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    # # 添加全局最大速度标记到图例
    # if max_info:
    #     star_patch = mpatches.Patch(color='red', label='全局最大速度点')
    #     handles, labels = ax.get_legend_handles_labels()
    #     handles.append(star_patch)
    #     labels.append('全局最大速度点')
    #     ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=11, framealpha=0.9, shadow=True)
    
    # 设置坐标轴范围，为注释留出更多空间
    time_range = df['time'].max() - df['time'].min()
    vel_range = df['velocity_m_s'].max() - df['velocity_m_s'].min()
    ax.set_xlim(df['time'].min(), df['time'].max())
    ax.set_ylim(df['velocity_m_s'].min() - vel_range * 0.15, df['velocity_m_s'].max() + vel_range * 0.25)
    
#     # 添加统计信息文本框
#     stats_text = f"""统计信息:
# 搜索区间: [{df['time'].min():.1f}, {df['time'].max():.1f}]s
# 时间步长: 0.001s
# 总数据点: {len(df)} 个
# 全局最大速度: {max_info['velocity_m_s']:.6f} m/s"""
    
#     ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
#             verticalalignment='top', fontsize=10,
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 时间-速度曲线图已保存到: {save_path}")
    else:
        plt.show()
    
    return fig, ax

def main():
    """主函数"""
    try:
        df, max_info = find_max_velocity_in_interval()
        
        print(f"\n✅ 搜索任务完成！")
        if max_info:
            print(f"🏆 全局最大速度: {max_info['velocity_m_s']:.6f} m/s")
            print(f"🕐 发生时间: {max_info['time']:.1f}s")
            print(f"📍 节点位置: {max_info['node_name']}")
        
        # 生成时间-速度曲线图
        plot_save_path = current_dir / "velocity_time_curves.png"
        plot_velocity_curves(df, max_info, save_path=plot_save_path)
        
        print(f"\n🎨 可视化完成！所有文件已保存到 {current_dir}")
        
    except Exception as e:
        print(f"\n❌ 程序执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
