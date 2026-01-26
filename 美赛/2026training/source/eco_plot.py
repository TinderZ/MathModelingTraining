import matplotlib.pyplot as plt
import numpy as np

def plot_biomass(t, N, species_params, save_path=None, title_suffix=""):
    plt.figure(figsize=(10, 5))
    for i, sp in enumerate(species_params):
        plt.plot(t, N[:, i], label=sp["name"])
    plt.title(f"Species Biomass N_i(t) {title_suffix}")
    plt.xlabel("Week")
    plt.ylabel("Biomass")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_carrying_capacity(t, K, species_params, save_path=None):
    plt.figure(figsize=(10, 5))
    for i, sp in enumerate(species_params):
        plt.plot(t, K[:, i], label=f'{sp["name"]} K(t)')
    plt.title("Dynamic Carrying Capacity K_i(t)")
    plt.xlabel("Week")
    plt.ylabel("K")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_precipitation_and_drought(t, P, C, save_path=None):
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(t, P, color="tab:blue", label="Precipitation P(t)")
    ax1.set_xlabel("Week")
    ax1.set_ylabel("P(t)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(t, C, color="tab:red", label="Drought Counter C(t)")
    ax2.set_ylabel("C(t)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Precipitation and Drought Counter")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_advanced_metrics(t, metrics, save_path=None):
    """
    绘制高级指标随时间的变化:
    1. Shannon 多样性 H(t)
    2. 平均竞争压力 Pi(t)
    3. 动态稳定性 lambda_max(t)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Shannon
    axes[0].plot(t, metrics["shannon_H_t"], color="purple")
    axes[0].set_title("Shannon Diversity H(t)")
    axes[0].set_ylabel("H(t)")
    axes[0].grid(True, alpha=0.3)
    
    # Competition Pressure
    axes[1].plot(t, metrics["competition_pressure_t"], color="orange")
    axes[1].axhline(y=1.0, color="red", linestyle="--", label="Threshold=1.0")
    axes[1].set_title("Average Competition Pressure Pi(t)")
    axes[1].set_ylabel("Pi(t)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Stability
    axes[2].plot(t, metrics["stability_lambda_t"], color="green")
    axes[2].axhline(y=0.0, color="red", linestyle="--", label="Stability Boundary (0)")
    axes[2].set_title("Dynamic Stability lambda_max(t)")
    axes[2].set_ylabel("Max Re(lambda)")
    axes[2].set_xlabel("Week")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_comparison_metrics(t, results_list, save_prefix="comparison"):
    """
    将多次实验的高级指标画在同一张图上对比
    results_list: list of dict, each dict contains:
        - 'n': species count
        - 'metrics': metrics dict from compute_metrics
    """
    # 1. Shannon Comparison
    plt.figure(figsize=(10, 6))
    for res in results_list:
        n = res['n']
        metrics = res['metrics']
        plt.plot(t, metrics["shannon_H_t"], label=f"n={n}")
    plt.title("Shannon Diversity H(t) Comparison")
    plt.xlabel("Week")
    plt.ylabel("H(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_shannon.png")
    plt.close()

    # 2. Competition Pressure Comparison
    plt.figure(figsize=(10, 6))
    for res in results_list:
        n = res['n']
        metrics = res['metrics']
        plt.plot(t, metrics["competition_pressure_t"], label=f"n={n}")
    plt.axhline(y=1.0, color="red", linestyle="--", label="Threshold=1.0")
    plt.title("Competition Pressure Pi(t) Comparison")
    plt.xlabel("Week")
    plt.ylabel("Pi(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_competition.png")
    plt.close()

    # 3. Stability Comparison
    plt.figure(figsize=(10, 6))
    for res in results_list:
        n = res['n']
        metrics = res['metrics']
        plt.plot(t, metrics["stability_lambda_t"], label=f"n={n}")
    plt.axhline(y=0.0, color="red", linestyle="--", label="Stability Boundary (0)")
    plt.title("Dynamic Stability lambda_max(t) Comparison")
    plt.xlabel("Week")
    plt.ylabel("Max Re(lambda)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_stability.png")
    plt.close()
