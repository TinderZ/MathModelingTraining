#!/usr/bin/env python3
"""
速度热力图可视化程序
绘制不同时间点的板凳龙位置，用速度确定颜色
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from position_velocity_calculator import PositionVelocityCalculator
from t4simulate import DragonRectangleSimulation

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class VelocityHeatmapVisualizer:
    """速度热力图可视化类"""
    
    def __init__(self, simulation: DragonRectangleSimulation):
        self.sim = simulation
        self.calc = PositionVelocityCalculator(simulation)
        
    def create_heatmap_for_time(self, time_point, ax, title_suffix=""):
        """
        为指定时间点创建速度热力图
        
        Args:
            time_point: 时间点（秒）
            ax: matplotlib轴对象
            title_suffix: 标题后缀
        """
        # 更新仿真到指定时间
        self.sim.time = time_point
        self.sim.update_all_positions()
        
        # 计算速度
        velocities_cm_s = self.calc.calculate_velocities_at_time()
        velocities_m_s = velocities_cm_s / 100.0  # 转换为m/s
        
        # 获取位置数据（转换为米）
        positions = np.array(self.sim.positions) / 100.0  # cm转m
        
        # 创建颜色映射，基于实际的最大最小速度
        vmin = np.min(velocities_m_s)
        vmax = np.max(velocities_m_s)
        # 稍微扩展范围以避免颜色过于极端
        v_range = vmax - vmin
        if v_range > 0:
            vmin = vmin - v_range * 0.02
            vmax = vmax + v_range * 0.02
        else:
            # 如果所有速度相同，设置一个小范围
            vmin = vmin - 0.01
            vmax = vmax + 0.01
        
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.plasma  # 使用plasma颜色映射
        
        # 绘制节点间的连线（木棍），颜色由两端速度插值确定
        segments = []
        colors = []
        
        for i in range(len(positions) - 1):
            # 线段两端点
            p1 = positions[i]
            p2 = positions[i + 1]
            segments.append([p1, p2])
            
            # 线段颜色由两端速度的平均值确定
            avg_velocity = (velocities_m_s[i] + velocities_m_s[i + 1]) / 2.0
            colors.append(avg_velocity)
        
        # 创建线段集合
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2, alpha=0.8)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)
        
        # 绘制节点点，颜色由速度确定
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=velocities_m_s, cmap=cmap, norm=norm,
                           s=20, edgecolors='black', linewidths=0.5, 
                           zorder=5, alpha=0.9)
        
        # 特别标记龙头
        ax.scatter(positions[0, 0], positions[0, 1], 
                  c='red', s=100, marker='*', 
                  edgecolors='black', linewidths=1,
                  zorder=10, label='龙头')
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f't = {time_point}s 板凳龙速度分布{title_suffix}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 添加背景轨迹
        self._add_background_trajectory(ax)
        
        return scatter, norm, vmax
    
    def _add_background_trajectory(self, ax):
        """添加背景轨迹"""
        # 绘制螺旋线
        theta_range = np.linspace(0, max(self.sim.initial_theta, 40), 1000)
        r_range = self.sim.spiral_coeff * theta_range
        x_spiral = r_range * np.cos(theta_range) / 100.0  # 转换为米
        y_spiral = r_range * np.sin(theta_range) / 100.0
        ax.plot(x_spiral, y_spiral, '--', color='gray', linewidth=1, alpha=0.5, label='盘入螺旋轨迹')
        ax.plot(-x_spiral, -y_spiral, '--', color='gold', linewidth=1, alpha=0.5, label='盘出螺旋轨迹')
        
        # 绘制掉头空间圆
        circle = Circle((0, 0), 4.5, color='purple', fill=False, 
                       linestyle='-.', linewidth=1, alpha=0.5, label='掉头空间(半径4.5m)')
        ax.add_patch(circle)
        
        # 标记关键点
        D_m = self.sim.D / 100.0  # 转换为米
        E_m = self.sim.E / 100.0
        ax.plot(D_m[0], D_m[1], 'go', markersize=6, alpha=0.7, label=f'D点')
        ax.plot(E_m[0], E_m[1], 'mo', markersize=6, alpha=0.7, label=f'E点')
    
    def _create_heatmap_for_time_with_norm(self, time_point, ax, norm, title_suffix=""):
        """
        为指定时间点创建速度热力图，使用指定的颜色映射标准化
        
        Args:
            time_point: 时间点（秒）
            ax: matplotlib轴对象
            norm: 颜色映射标准化对象
            title_suffix: 标题后缀
        """
        # 更新仿真到指定时间
        self.sim.time = time_point
        self.sim.update_all_positions()
        
        # 计算速度
        velocities_cm_s = self.calc.calculate_velocities_at_time()
        velocities_m_s = velocities_cm_s / 100.0  # 转换为m/s
        
        # 获取位置数据（转换为米）
        positions = np.array(self.sim.positions) / 100.0  # cm转m
        
        # 使用指定的颜色映射
        cmap = plt.cm.plasma
        
        # 绘制节点间的连线（木棍），颜色由两端速度插值确定
        segments = []
        colors = []
        
        for i in range(len(positions) - 1):
            # 线段两端点
            p1 = positions[i]
            p2 = positions[i + 1]
            segments.append([p1, p2])
            
            # 线段颜色由两端速度的平均值确定
            avg_velocity = (velocities_m_s[i] + velocities_m_s[i + 1]) / 2.0
            colors.append(avg_velocity)
        
        # 创建线段集合
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2, alpha=0.8)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)
        
        # 绘制节点点，颜色由速度确定
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=velocities_m_s, cmap=cmap, norm=norm,
                           s=20, edgecolors='black', linewidths=0.5, 
                           zorder=5, alpha=0.9)
        
        # 特别标记龙头
        ax.scatter(positions[0, 0], positions[0, 1], 
                  c='red', s=100, marker='*', 
                  edgecolors='black', linewidths=1,
                  zorder=10, label='龙头')
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f't = {time_point}s 板凳龙速度分布{title_suffix}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 添加背景轨迹
        self._add_background_trajectory(ax)
        
        return scatter
    
    def _create_heatmap_for_time_with_norm_custom(self, time_point, ax, norm, title_suffix=""):
        """
        为指定时间点创建速度热力图，使用指定的颜色映射标准化（不设置标题）
        
        Args:
            time_point: 时间点（秒）
            ax: matplotlib轴对象
            norm: 颜色映射标准化对象
            title_suffix: 标题后缀
        """
        # 更新仿真到指定时间
        self.sim.time = time_point
        self.sim.update_all_positions()
        
        # 计算速度
        velocities_cm_s = self.calc.calculate_velocities_at_time()
        velocities_m_s = velocities_cm_s / 100.0  # 转换为m/s
        
        # 获取位置数据（转换为米）
        positions = np.array(self.sim.positions) / 100.0  # cm转m
        
        # 使用指定的颜色映射
        cmap = plt.cm.plasma
        
        # 绘制节点间的连线（木棍），颜色由两端速度插值确定
        segments = []
        colors = []
        
        for i in range(len(positions) - 1):
            # 线段两端点
            p1 = positions[i]
            p2 = positions[i + 1]
            segments.append([p1, p2])
            
            # 线段颜色由两端速度的平均值确定
            avg_velocity = (velocities_m_s[i] + velocities_m_s[i + 1]) / 2.0
            colors.append(avg_velocity)
        
        # 创建线段集合
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2, alpha=0.8)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)
        
        # 绘制节点点，颜色由速度确定
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=velocities_m_s, cmap=cmap, norm=norm,
                           s=20, edgecolors='black', linewidths=0.5, 
                           zorder=5, alpha=0.9)
        
        # 特别标记龙头
        ax.scatter(positions[0, 0], positions[0, 1], 
                  c='red', s=100, marker='*', 
                  edgecolors='black', linewidths=1,
                  zorder=10, label='龙头')
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        # 不设置标题，由调用方处理
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 添加背景轨迹
        self._add_background_trajectory(ax)
        
        return scatter
    
    def create_multi_time_heatmap(self, time_points=None, save_path=None):
        """
        创建多个时间点的速度热力图
        
        Args:
            time_points: 时间点列表，默认为[-100, -50, 0, 50, 100]
            save_path: 保存路径，如果为None则显示图形
        """
        if time_points is None:
            time_points = [-100, -50, 0, 50, 100]
        
        # 预先计算所有时间点的速度范围，确保统一的颜色映射
        all_velocities = []
        for time_point in time_points:
            self.sim.time = time_point
            self.sim.update_all_positions()
            velocities_cm_s = self.calc.calculate_velocities_at_time()
            velocities_m_s = velocities_cm_s / 100.0
            all_velocities.extend(velocities_m_s)
        
        # 计算全局速度范围
        global_vmin_original = np.min(all_velocities)
        global_vmax_original = np.max(all_velocities)
        
        # 设置颜色条的实际范围
        global_vmin = global_vmin_original * 0.8
        global_vmax = global_vmax_original
        
        global_norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
        
        # 创建子图，为颜色条预留空间
        n_times = len(time_points)
        cols = 2
        rows = 2
        
        # 创建图形，为颜色条预留右侧空间
        fig = plt.figure(figsize=(6*cols + 1.5, 5*rows))
        
        # 创建网格布局，为颜色条预留空间
        gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1, 1, 0.1], 
                             hspace=0.4, wspace=0.3, 
                             left=0.08, right=0.85, top=0.95, bottom=0.15)
        
        axes = []
        for i in range(rows):
            for j in range(cols):
                if len(axes) < n_times:
                    ax = fig.add_subplot(gs[i, j])
                    axes.append(ax)
        
        # 为每个时间点创建热力图，使用统一的颜色映射
        scatters = []
        
        for i, time_point in enumerate(time_points):
            if i < len(axes):
                scatter = self._create_heatmap_for_time_with_norm_custom(time_point, axes[i], global_norm)
                scatters.append(scatter)
                
                # 将标题放在图片下方
                axes[i].set_title('')  # 清除原标题
                axes[i].text(0.5, -0.15, f't = {time_point}s 板凳龙速度分布', 
                           transform=axes[i].transAxes, 
                           ha='center', va='top', fontsize=12, weight='bold')
        
        # 添加颜色条到最右边
        if scatters:
            # 创建颜色条的位置
            cbar_ax = fig.add_subplot(gs[:, -1])
            cbar = fig.colorbar(scatters[0], cax=cbar_ax)
            cbar.set_label('速度 (m/s)', fontsize=12)
            # 设置颜色条的范围和标签
            # 颜色条范围为[global_vmin*0.8, global_vmax/0.8]
            # 但标签在v=1, 1.2, 1.4处标记
            cbar.set_ticks([1.0, 1.2, 1.4])
            cbar.set_ticklabels(['1.0', '1.2', '1.4'])
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存到: {save_path}")
        else:
            plt.show()
        
        return fig, axes
    
    def create_single_large_heatmap(self, time_point, save_path=None):
        """
        创建单个时间点的大尺寸详细热力图
        
        Args:
            time_point: 时间点（秒）
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter, norm, vmax = self.create_heatmap_for_time(time_point, ax, " (详细视图)")
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('速度 (m/s)', fontsize=14)
        
        # 添加图例
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
        
        # 添加统计信息
        self.sim.time = time_point
        self.sim.update_all_positions()
        velocities_cm_s = self.calc.calculate_velocities_at_time()
        velocities_m_s = velocities_cm_s / 100.0
        
        # 设置颜色条的刻度，显示实际的最大最小速度
        vmin_actual = np.min(velocities_m_s)
        vmax_actual = np.max(velocities_m_s)
        cbar.set_ticks([vmin_actual, (vmin_actual + vmax_actual) / 2, vmax_actual])
        cbar.set_ticklabels([f'{vmin_actual:.3f}', f'{(vmin_actual + vmax_actual) / 2:.3f}', f'{vmax_actual:.3f}'])
        
        stats_text = f"""统计信息:
最大速度: {np.max(velocities_m_s):.3f} m/s
最小速度: {np.min(velocities_m_s):.3f} m/s
平均速度: {np.mean(velocities_m_s):.3f} m/s
"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细热力图已保存到: {save_path}")
        else:
            plt.show()
        
        return fig, ax

def main():
    """主函数"""
    print("="*60)
    print("板凳龙速度热力图可视化")
    print("="*60)
    
    # 创建仿真实例
    sim = DragonRectangleSimulation(pitch=170.0, debug=False)
    
    # 创建可视化器
    visualizer = VelocityHeatmapVisualizer(sim)
    
    # 时间点
    test_times = [ 0, 14.5, 50, 100]
    
    print("正在生成多时间点热力图...")
    # 创建多时间点热力图
    visualizer.create_multi_time_heatmap(
        time_points=test_times,
        save_path="dragon_velocity_heatmap_multi.png"
    )
    
    # # 为每个关键时间点创建详细热力图
    # key_times = [14.5]  # 选择两个关键时间点
    # for t in key_times:
    #     print(f"正在生成 t={t}s 的详细热力图...")
    #     visualizer.create_single_large_heatmap(
    #         time_point=t,
    #         save_path=f"dragon_velocity_heatmap_t{t:+d}s.png"
    #     )

if __name__ == "__main__":
    main()
