import numpy as np

from eco_data import compute_drought_counter


def monod_response(S, h):
    """
    基础水分响应 F_i(S(t))，Monod 方程：
    F_i = S / (S + h)
    """
    return S / (S + h)


def water_factor(S, C, h, L, beta):
    """
    水分综合因子 W_i(t)：
    - 若 C(t) <= L，W_i = F_i(S)
    - 若 C(t) > L，W_i = F_i(S) * exp(-beta * (C(t) - L))
    """
    F = monod_response(S, h)
    W = np.where(C <= L, F, F * np.exp(-beta * (C - L)))
    return W


def seasonal_factor(t, mu, sigma):
    """
    季节性因子 G_i(t)：
    G_i = exp(-(t - mu)**2 / (2 * sigma**2))
    """
    tt = t % 52
    return 0.7 + 0.3 * np.exp(-((tt - mu)**2) / (2 * sigma**2))

def simulate_lv_dynamic_K(
    total_weeks,
    species_params,
    alpha,
    N0,
    P,
    P_min=1.0,
    gamma=0.9,
    dt=1.0,
    eps=1e-6,
):
    """
    使用欧拉法求解带动态环境容量 K_i(t) 的 Lotka-Volterra 竞争模型。

    微分方程：
    dN_i/dt = r_i * N_i * (1 - sum_j(alpha_ij * N_j) / (K_i(t) + eps))

    动态环境容量：
    K_i(t) = K_max,i * G_i(t) * W_i(t)
    注意：按要求 G_i(t) 先置为 1。
    """
    num_species = len(species_params)
    N = np.zeros((total_weeks, num_species), dtype=float)
    K = np.zeros((total_weeks, num_species), dtype=float)

    # 初始化
    N[0, :] = N0

    # 计算干旱计数器
    C = compute_drought_counter(P, P_min)

    # 有效土壤水分 S(t) = P(t) + gamma * S(t-1)
    S = np.zeros_like(P, dtype=float)
    if len(P) > 0:
        S[0] = P[0]
        for t in range(1, len(P)):
            S[t] = P[t] + gamma * S[t - 1]

    epsilon = 0.2
    for t in range(total_weeks):
        # 当前时刻的水分因子
        for i in range(num_species):
            p = species_params[i]
            W_i = water_factor(S[t], C[t], p["h"], p["L"], p["beta"])
            G_i = 1.0  # 按要求暂时舍去季节物候项
            G_i = seasonal_factor(t, p["mu"], p["sigma"])
            K[t, i] = p["K_max"] * G_i * W_i
            #K[t, i] = epsilon * p["K_max"] + (1 - epsilon) * p["K_max"] * G_i * W_i

        if t == total_weeks - 1:
            break

        # 计算欧拉步进
        for i in range(num_species):
            r_i = species_params[i]["r"]
            competition = np.dot(alpha[i, :], N[t, :])
            growth = r_i * N[t, i] * (1.0 - competition / (K[t, i] + eps))
            N[t + 1, i] = max(N[t, i] + dt * growth, 0.0)

    return N, K, C
