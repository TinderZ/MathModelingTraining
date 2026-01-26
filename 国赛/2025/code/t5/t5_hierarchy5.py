import numpy as np
import pyswarms as ps
import time
from t5_common import *

if __name__ == "__main__":
    bounds, all_bounds = get_drone_bounds()
    bounds_low, bounds_high = bounds
    
    STD_DEV_DIVISOR = 20.0
    PRIMARY_DRONES = [0]  # 第5层: 优化 FY1
    SECONDARY_DRONES = [1, 2, 3, 4]
    N_PRIMARY_SOLUTIONS = 100
    N_SECONDARY_SOLUTIONS_PER_DRONE = 50
    N_COMBINATIONS_PER_PRIMARY = 10
    N_PARTICLES = N_PRIMARY_SOLUTIONS * N_COMBINATIONS_PER_PRIMARY

    print("开始生成分层优化高质量初始粒子 (第5层: FY1)...")

    drone_solutions = {}
    for i in PRIMARY_DRONES:
        solutions = generate_valid_solutions_for_drone_t5(i, all_bounds[i], N_PRIMARY_SOLUTIONS)
        if solutions.shape[0] == 0:
            raise ValueError(f"无法为主要目标无人机 FY{i+1} 找到任何可行解。")
        drone_solutions[i] = solutions

    for i in SECONDARY_DRONES:
        solutions = generate_valid_solutions_for_drone_t5_normal(i, all_bounds[i], N_SECONDARY_SOLUTIONS_PER_DRONE, OPTIMAL_PARAMS_REF[i], STD_DEV_DIVISOR)
        if solutions.shape[0] == 0:
            solutions = generate_valid_solutions_for_drone_t5(i, all_bounds[i], N_SECONDARY_SOLUTIONS_PER_DRONE)
        if solutions.shape[0] == 0:
            raise ValueError(f"无法为次要目标无人机 FY{i+1} 找到任何可行解。")
        drone_solutions[i] = solutions

    init_pos_list = []
    primary_solutions_fy1 = drone_solutions[0]

    for prim_sol in primary_solutions_fy1:
        for _ in range(N_COMBINATIONS_PER_PRIMARY):
            sec_sol_fy2 = drone_solutions[1][np.random.randint(0, len(drone_solutions[1]))]
            sec_sol_fy3 = drone_solutions[2][np.random.randint(0, len(drone_solutions[2]))]
            sec_sol_fy4 = drone_solutions[3][np.random.randint(0, len(drone_solutions[3]))]
            sec_sol_fy5 = drone_solutions[4][np.random.randint(0, len(drone_solutions[4]))]
            particle = np.concatenate([prim_sol, sec_sol_fy2, sec_sol_fy3, sec_sol_fy4, sec_sol_fy5])
            init_pos_list.append(particle)

    init_pos = np.array(init_pos_list)
    
    print("开始为5架无人机对抗3枚导弹进行优化...")
    start_time = time.time()
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(n_particles=N_PARTICLES, dimensions=40, options=options, bounds=bounds, init_pos=init_pos)
    best_cost, best_params = optimizer.optimize(calculate_occlusion_time_parallel, iters=100, verbose=True)
    
    end_time = time.time()
    print("优化完成！")
    max_occlusion_time = -best_cost
    print("找到的最佳结果:")
    params_per_drone = best_params.reshape(5, 8)
    
    for i in range(5):
        a, v, t_take1, t_fall1, t_take2, t_fall2, t_take3, t_fall3 = params_per_drone[i]
        print(f"--- 无人机 FY{i+1} ---")
        print(f"飞行方向角: {np.rad2deg(a):.4f} 度")
        print(f"飞行速度: {v:.4f} m/s")
        print(f"烟幕弹 1: 投放时间 = {t_take1:.4f} s, 下落时间 = {t_fall1:.4f} s")
        print(f"烟幕弹 2: 投放时间 = {t_take2:.4f} s, 下落时间 = {t_fall2:.4f} s")
        print(f"烟幕弹 3: 投放时间 = {t_take3:.4f} s, 下落时间 = {t_fall3:.4f} s")

    print(f"最大协同遮蔽时间: {max_occlusion_time:.4f} 秒")