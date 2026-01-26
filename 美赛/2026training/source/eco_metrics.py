import numpy as np


def compute_metrics(N, K, species_params, alpha, extinction_threshold=0.0, endangered_ratio=0.2):
    """
    计算指标：
    - 灭绝数量（是否出现 N <= extinction_threshold）
    - 平均濒危周数（N < 20% K_max）
    - 平均方差（各物种 N 的时间序列方差）
    - 物种异步性 phi
    - Shannon 多样性 H(t) 及平均值
    - 竞争压力指数 Pi(t) 及平均值
    - 动态稳定性 (最大特征值实部) lambda_max(t) 及平均值
    """
    # 提取参数
    K_max = np.array([sp["K_max"] for sp in species_params], dtype=float)
    r = np.array([sp["r"] for sp in species_params], dtype=float)
    total_weeks, total_species = N.shape
    
    # --- 1. 基础生存指标 ---
    extinct_flags = (N <= extinction_threshold).any(axis=0)
    extinct_count = int(extinct_flags.sum())
    surviving_count = total_species - extinct_count

    endangered_thresholds = endangered_ratio * K_max
    endangered_weeks = (N < endangered_thresholds).sum(axis=0)
    avg_endangered_weeks = float(np.mean(endangered_weeks))

    variances = np.var(N, axis=0)
    avg_variance = float(np.mean(variances))

    # --- 2. 物种异步性 (Portfolio effect) ---
    # phi = Var(sum(Ni)) / sum(Var(Ni))
    total_biomass_ts = N.sum(axis=1)
    var_total_biomass = np.var(total_biomass_ts)
    sum_var_species = np.sum(variances)
    
    if sum_var_species > 0:
        asynchrony_phi = var_total_biomass / sum_var_species
    else:
        asynchrony_phi = 1.0  # 如果没有变异，视为同步或无意义

    # --- 3. Shannon 多样性 ---
    # H(t) = - sum( pi * ln(pi) )
    # pi = Ni / sum(Ni)
    # 避免除零和 log(0)
    
    # 加上微小量避免除零
    total_biomass_per_step = N.sum(axis=1, keepdims=True) + 1e-9
    p = N / total_biomass_per_step
    
    # 计算 -p * ln(p)，注意 p=0 时结果为 0
    plnp = np.zeros_like(p)
    mask = p > 0
    plnp[mask] = p[mask] * np.log(p[mask])
    
    shannon_H = -np.sum(plnp, axis=1)
    avg_shannon = float(np.mean(shannon_H))

    # --- 4. 竞争压力指数 (Competition Pressure) ---
    # Pi_i(t) = sum_j(alpha_ij * Nj(t)) / Ki(t)
    # Pi(t) = mean(Pi_i(t))
    
    # alpha: (n, n), N.T: (n, t) -> competition terms: (n, t)
    # competition_load[i, t] = sum_j alpha_ij * Nj(t)
    competition_load = alpha @ N.T  # shape (n, t)
    
    # Ki(t): (t, n) -> need (n, t)
    K_safe = K.T.copy()
    K_safe[K_safe < 1e-6] = 1e-6 # 避免除以 0
    
    Pi_matrix = competition_load / K_safe # shape (n, t)
    Pi_t = np.mean(Pi_matrix, axis=0) # Average over species for each time step -> (t,)
    
    avg_competition_pressure = float(np.mean(Pi_t))
    prob_overload = float(np.mean(Pi_t > 1.0)) # Pr(Pi > 1)

    # --- 5. 动态稳定性 (Dynamic Stability / Lyapunov) ---
    # Jacobian J_ij(t)
    # J_ij = ri * [ (1 - Pi_i) * delta_ij - (Ni * alpha_ij) / Ki ]
    # Calculate max real eigenvalue at each step
    
    lambda_max_t = np.zeros(total_weeks)
    
    for t in range(total_weeks):
        # Construct Jacobian for time t
        # Pi_i for this t is Pi_matrix[:, t]
        # N_i for this t is N[t, :]
        # K_i for this t is K[t, :] -> K_safe[:, t]
        
        Pi_vec = Pi_matrix[:, t]
        N_vec = N[t, :]
        K_vec = K_safe[:, t]
        
        # Term 1: diag(r_i * (1 - Pi_i))
        term1 = np.diag(r * (1 - Pi_vec))
        
        # Term 2: r_i * N_i * alpha_ij / K_i
        # (N_i / K_i) can be precomputed as vector
        # factor_i = r_i * N_i / K_i
        # Term 2_ij = factor_i * alpha_ij
        factor = r * N_vec / K_vec
        term2 = factor[:, np.newaxis] * alpha  # broadcasting factor to rows
        
        J = term1 - term2
        
        # Eigenvalues
        try:
            eigvals = np.linalg.eigvals(J)
            max_real_part = np.max(eigvals.real)
            lambda_max_t[t] = max_real_part
        except:
            lambda_max_t[t] = np.nan

    # Remove NaNs if any
    valid_lambdas = lambda_max_t[~np.isnan(lambda_max_t)]
    if len(valid_lambdas) > 0:
        avg_lambda_max = float(np.mean(valid_lambdas))
        prob_unstable = float(np.mean(valid_lambdas > 0))
    else:
        avg_lambda_max = 0.0
        prob_unstable = 0.0

    return {
        "total_species": total_species,
        "extinct_count": extinct_count,
        "surviving_count": surviving_count,
        "avg_endangered_weeks": avg_endangered_weeks,
        "avg_variance": avg_variance,
        "endangered_weeks_per_species": endangered_weeks.tolist(),
        "variance_per_species": variances.tolist(),
        # New metrics
        "asynchrony_phi": asynchrony_phi,
        "shannon_H_t": shannon_H,
        "avg_shannon": avg_shannon,
        "competition_pressure_t": Pi_t,
        "avg_competition_pressure": avg_competition_pressure,
        "prob_overload": prob_overload,
        "stability_lambda_t": lambda_max_t,
        "avg_stability": avg_lambda_max,
        "prob_unstable": prob_unstable
    }
