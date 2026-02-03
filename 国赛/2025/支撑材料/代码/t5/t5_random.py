import numpy as np
import time
from t5_common import *

if __name__ == "__main__":
    bounds, _ = get_drone_bounds()
    bounds_low, bounds_high = bounds

    best_params_mean = np.array([
        np.deg2rad(179.5824), 139.5909, 0.2246, 3.7469, 4.2000, 5.4670, 6.1853, 7.7078,
        np.deg2rad(295.3677), 99.4268, 9.2210, 1.6224, 11.8339, 8.9548, 20.7638, 2.2823,
        np.deg2rad(87.5943), 92.8446, 27.7529, 2.7758, 29.1748, 3.7963, 49.2416, 7.3458,
        np.deg2rad(275.8299), 114.2396, 3.8243, 10.5066, 9.4303, 10.6933, 23.1460, 12.0854,
        np.deg2rad(109.9545), 131.3048, 12.7298, 3.7228, 28.4380, 7.9242, 33.6368, 4.5380,
    ])

    std_devs = (bounds_high - bounds_low) / 30.0
    N_ITERATIONS = 100
    results = []
    
    print(f"开始随机搜索 {N_ITERATIONS} 次...")
    start_time = time.time()

    for i in range(N_ITERATIONS):
        random_params = np.random.normal(loc=best_params_mean, scale=std_devs, size=40)
        random_params = np.clip(random_params, bounds_low, bounds_high)
        cost = calculate_occlusion_time_single(random_params)
        occlusion_time = -cost
        if occlusion_time < -100:
             occlusion_time = 0.0
        results.append(occlusion_time)

    end_time = time.time()
    print("搜索完成！")
    print("统计信息:")
    print(f"最大遮蔽时间: {np.max(results):.4f} 秒")
    print(f"最小遮蔽时间: {np.min(results):.4f} 秒")
    print(f"平均遮蔽时间: {np.mean(results):.4f} 秒")