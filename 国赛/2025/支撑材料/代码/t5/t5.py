import numpy as np
import pyswarms as ps
import time
from t5_common import *

if __name__ == "__main__":
    bounds, all_bounds = get_drone_bounds()
    bounds_low, bounds_high = bounds
    
    N_PARTICLES = 3000
    print("开始生成高质量初始粒子...")
    
    n_valid_per_drone = 30
    drone_solutions = []
    for i in range(5):
        solutions = generate_valid_solutions_for_drone_t5(i, all_bounds[i], n_valid_per_drone)
        if solutions.shape[0] == 0:
            raise ValueError(f"无法为无人机 FY{i+1} 找到任何可行解。")
        drone_solutions.append(solutions)

    n_good_particles_target = 2000
    good_particles_list = []
    for _ in range(n_good_particles_target):
        particle_parts = [sol[np.random.randint(0, len(sol))] for sol in drone_solutions]
        particle = np.concatenate(particle_parts)
        good_particles_list.append(particle)
    good_particles = np.array(good_particles_list)
    n_random_particles = N_PARTICLES - n_good_particles_target
    random_particles = np.random.uniform(low=bounds_low, high=bounds_high, size=(n_random_particles, 40))
    init_pos = np.vstack([good_particles, random_particles])
    
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