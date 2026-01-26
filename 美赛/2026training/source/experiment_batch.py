import os
import numpy as np
import matplotlib.pyplot as plt

from eco_config import load_species_config
from eco_data import generate_precipitation
from eco_metrics import compute_metrics
from eco_model import simulate_lv_dynamic_K
from eco_plot import (
    plot_biomass,
    plot_comparison_metrics
)

def run_batch_experiments():
    # --------------------------
    # 1) 通用参数设置
    # --------------------------
    years = 5
    weeks_per_year = 52
    total_weeks = years * weeks_per_year
    
    # 输出目录
    output_dir = "results_batch"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 统一降水序列，保证实验可比性
    print("生成统一降水序列...")
    P = generate_precipitation(total_weeks)
    
    # 实验配置
    n_values = [2, 3, 4, 5, 6, 7]
    results_list = []
    t = np.arange(total_weeks)

    print(f"开始批量实验: n = {n_values}")

    # --------------------------
    # 2) 循环运行实验
    # --------------------------
    for n in n_values:
        print(f"\n--- Running experiment for n={n} ---")
        
        # 加载配置
        try:
            species_params, alpha, N0 = load_species_config(n)
        except ValueError as e:
            print(f"Error loading config for n={n}: {e}")
            continue

        # 运行模拟
        N, K, C = simulate_lv_dynamic_K(
            total_weeks=total_weeks,
            species_params=species_params,
            alpha=alpha,
            N0=N0,
            P=P,
            P_min=1.0,
            dt=1.0,
            eps=1e-6,
        )

        # 计算指标
        metrics = compute_metrics(N, K, species_params, alpha)
        
        # 保存结果用于后续对比
        results_list.append({
            "n": n,
            "metrics": metrics,
            "N": N,
            "species_params": species_params
        })

        # 保存单次实验的生物总量图
        plot_filename = os.path.join(output_dir, f"biomass_n_{n}.png")
        plot_biomass(t, N, species_params, save_path=plot_filename, title_suffix=f"(n={n})")
        print(f"Saved biomass plot to {plot_filename}")
        
        # 打印简要指标
        print(f"  Avg Shannon: {metrics['avg_shannon']:.4f}")
        print(f"  Asynchrony (phi): {metrics['asynchrony_phi']:.4f}")
        print(f"  Avg Stability: {metrics['avg_stability']:.4f}")

    # --------------------------
    # 3) 绘制对比图
    # --------------------------
    print("\n绘制对比图...")
    plot_comparison_metrics(t, results_list, save_prefix=os.path.join(output_dir, "comparison"))
    print(f"对比图已保存至 {output_dir} 目录")

if __name__ == "__main__":
    run_batch_experiments()
