import numpy as np
import matplotlib.pyplot as plt

from eco_config import load_species_config
from eco_data import generate_precipitation
from eco_metrics import compute_metrics
from eco_model import simulate_lv_dynamic_K
from eco_plot import (
    plot_biomass,
    plot_carrying_capacity,
    plot_precipitation_and_drought,
    plot_advanced_metrics,
)


def main():
    # --------------------------
    # 1) 参数设置
    # --------------------------
    years = 5
    weeks_per_year = 52
    total_weeks = years * weeks_per_year

    # 从配置文件读取物种参数与竞争系数
    # 输入 n（2-10），即可获取对应数量的物种参数
    n_str = input("请输入物种数量 n（2-10，默认 2）：").strip()
    n = int(n_str) if n_str else 2
    species_params, alpha, N0 = load_species_config(n)

    # 降水序列（包含干旱期）
    P = generate_precipitation(total_weeks)

    # --------------------------
    # 2) 求解
    # --------------------------
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

    t = np.arange(total_weeks)

    # --------------------------
    # 3) 计算指标
    # --------------------------
    metrics = compute_metrics(N, K, species_params, alpha)
    
    print("\n=== 指标汇总 ===")
    print(f"总物种数: {metrics['total_species']}")
    print(f"灭绝物种数: {metrics['extinct_count']}")
    print(f"存活物种数: {metrics['surviving_count']}")
    print(f"平均濒危周数(阈值=0.2*K_max): {metrics['avg_endangered_weeks']:.2f}")
    print(f"平均方差: {metrics['avg_variance']:.2f}")
    
    print("\n--- 高级生态指标 ---")
    print(f"物种异步性 phi (0~1, 越小越好): {metrics['asynchrony_phi']:.4f}")
    print(f"平均 Shannon 多样性: {metrics['avg_shannon']:.4f}")
    print(f"平均竞争压力 Pi: {metrics['avg_competition_pressure']:.4f}")
    print(f"超载竞争概率 Pr(Pi > 1): {metrics['prob_overload']:.4f}")
    print(f"平均动态稳定性 (Max Re(lambda)): {metrics['avg_stability']:.4f}")
    print(f"不稳定风险 Pr(lambda > 0): {metrics['prob_unstable']:.4f}")

    # --------------------------
    # 4) 可视化
    # --------------------------
    plot_biomass(t, N, species_params)
    plot_carrying_capacity(t, K, species_params)
    plot_precipitation_and_drought(t, P, C)
    
    # 新增：高级指标可视化
    plot_advanced_metrics(t, metrics)

    plt.show()


if __name__ == "__main__":
    main()
