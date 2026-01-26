import numpy as np


def generate_precipitation(
    total_weeks,
    # drought_start=20,
    # drought_end=30,
    seed=42,
    weeks_per_year=52,
    P_trans=None,
    p_total_values=None,
    e_min=2.0,
    e_max=6.0,
    precip_noise_sigma=5.0,
    return_details=False,
):
    """
    生成净水资源量 P(t)，包含年际马尔可夫状态与周尺度蒸散发。
    可选插入干旱期：drought_start ~ drought_end（含）。
    """
    rng = np.random.default_rng(seed)

    if P_trans is None:
        P_trans = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
            ],
            dtype=float,
        )
    if p_total_values is None:
        p_total_values = [400, 600, 800]

    total_years = int(np.ceil(total_weeks / weeks_per_year))

    # 年际状态生成（0=Dry, 1=Normal, 2=Wet）
    annual_states = [2]
    for _ in range(1, total_years):
        prev_s = annual_states[-1]
        curr_s = rng.choice([0, 1, 2], p=P_trans[prev_s])
        annual_states.append(curr_s)

    precip_raw = np.zeros(total_weeks, dtype=float)
    evap = np.zeros(total_weeks, dtype=float)
    p_net = np.zeros(total_weeks, dtype=float)

    # 预计算季节权重并归一化，使全年总量约等于 annual_val
    weeks = np.arange(1, weeks_per_year + 1)
    seasonal_weights = np.exp(-((weeks - 8) ** 2) / (2 * 12**2))
    seasonal_weights_sum = float(seasonal_weights.sum())

    for t in range(total_weeks):
        year_idx = t // weeks_per_year
        week_in_year = (t % weeks_per_year) + 1
        annual_val = p_total_values[annual_states[year_idx]]

        evap[t] = e_min + (e_max - e_min) * (1 - np.cos(2 * np.pi * week_in_year / 52)) / 2

        seasonal_weight = seasonal_weights[week_in_year - 1]
        precip_raw[t] = max(
            0.0,
            (annual_val * seasonal_weight / seasonal_weights_sum)
            + rng.normal(0.0, precip_noise_sigma),
        )


    p_net = np.maximum(0.0, precip_raw - evap)

    if return_details:
        return {
            "P": p_net,
            "precip_raw": precip_raw,
            "evap": evap,
            "annual_states": np.array(annual_states, dtype=int),
        }
    return p_net


def compute_drought_counter(P, P_min):
    """
    计算干旱计数器 C(t)：
    - 若 P(t) < P_min，C(t) = C(t-1) + 1
    - 若 P(t) >= P_min，C(t) = 0
    """
    C = np.zeros_like(P, dtype=int)
    if len(P) == 0:
        return C
    if P[0] < P_min:
        C[0] = 1
    for t in range(1, len(P)):
        if P[t] < P_min:
            C[t] = C[t - 1] + 1
        else:
            C[t] = 0
    return C
