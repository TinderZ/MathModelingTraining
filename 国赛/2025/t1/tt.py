import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def angle_BAC_2d(theta):
    """
    计算角BAC的大小 (C点限制在水平面z=0上)
    theta: 方位角 (0 to 2π)
    """
    # 圆上点C的坐标 (在XY平面上，z=0)
    x_c = np.cos(theta)
    y_c = np.sin(theta)
    z_c = 0  # 限制在水平面上
    C = np.array([x_c, y_c, z_c])
    
    # A和B的坐标
    A = np.array([1, 0, 2])
    B = np.array([1, 3, 2])
    
    # 向量CA和CB
    AB = A - B
    AC = A - C
    
    # 计算角BAC的余弦值
    dot_product = np.dot(AB, AC)
    norm_AB = np.linalg.norm(AB)
    norm_AC = np.linalg.norm(AC)
    
    # 避免除零错误
    if norm_AB == 0 or norm_AC == 0:
        return 0
    
    cos_angle = dot_product / (norm_AB * norm_AC)
    # 确保余弦值在[-1, 1]范围内
    cos_angle = np.clip(cos_angle, -1, 1)
    
    # 返回角度（弧度）
    angle = np.arccos(cos_angle)
    return angle



def find_max_angle_2d():
    """
    寻找使角BAC最大的点C (限制在水平面z=0上)
    """
    max_angle = 0
    best_theta = 0
    
    # 网格搜索
    theta_range = np.linspace(0, 2*np.pi, 100000)  # 增加精度
    
    for theta in theta_range:
        angle = angle_BAC_2d(theta)
        if angle > max_angle:
            max_angle = angle
            best_theta = theta
    
    # 计算最优点C的坐标 (在XY平面上)
    x_c = np.cos(best_theta)
    y_c = np.sin(best_theta)
    z_c = 0
    
    return np.array([x_c, y_c, z_c]), max_angle, best_theta


def analytical_solution_2d():
    """
    解析解方法 (C点限制在水平面z=0上)
    根据几何原理求解
    """
    A = np.array([1, 0, 2])
    B = np.array([1, 3, 2])
    
    print("=== 2D约束下的解析解分析 ===")
    print(f"A点: {A}")
    print(f"B点: {B}")
    
    # A和B在XY平面上的投影
    A_proj = np.array([A[0], A[1], 0])  # (1, 0, 0)
    B_proj = np.array([B[0], B[1], 0])  # (1, 3, 0)
    print(f"A在XY平面投影: {A_proj}")
    print(f"B在XY平面投影: {B_proj}")
    
    # 投影中点
    M_proj = (A_proj + B_proj) / 2
    print(f"投影中点M_proj: {M_proj}")
    
    # 从原点到投影中点的距离
    d_proj = np.linalg.norm(M_proj)
    print(f"原点到投影中点距离: {d_proj}")
    
    # 如果投影中点不在原点，则最优点在该方向上
    if d_proj > 1e-10:  # 避免除零
        C_optimal = M_proj / d_proj  # 单位化到单位圆上
    else:
        # 如果投影中点就在原点，需要其他方法
        C_optimal = np.array([1, 0, 0])  # 默认选择一个点
    
    print(f"最优点C坐标: {C_optimal}")
    print(f"验证|C|=1: {np.linalg.norm(C_optimal)}")
    print(f"验证z=0: {C_optimal[2]}")
    
    # 计算此时的角BAC
    CA = A - C_optimal
    CB = B - C_optimal
    cos_angle = np.dot(CA, CB) / (np.linalg.norm(CA) * np.linalg.norm(CB))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    return C_optimal, angle



if __name__ == "__main__":
    print("=== 三维空间中角BAC最大值问题求解 (C点限制在水平面z=0上) ===\n")
    
    # 问题描述
    print("问题：")
    print("- 圆C：圆心在原点(0,0,0)，半径为1，限制在水平面z=0上")
    print("- A点坐标：(1,0,2)")
    print("- B点坐标：(1,3,2)")
    print("- 求：当角∠BAC最大时，点C的坐标 (z=0)\n")
    
    # 方法1：2D数值优化
    print("方法1：2D网格搜索数值优化")
    C_numerical_2d, max_angle_num_2d, theta_opt_2d = find_max_angle_2d()
    print(f"最优点C坐标: ({C_numerical_2d[0]:.6f}, {C_numerical_2d[1]:.6f}, {C_numerical_2d[2]:.6f})")
    print(f"最大角BAC: {max_angle_num_2d:.6f} 弧度 = {np.degrees(max_angle_num_2d):.3f} 度")
    print(f"对应的角度参数: θ={theta_opt_2d:.6f}\n")
    
    
    
    print(angle_BAC_2d(1.5*np.pi))