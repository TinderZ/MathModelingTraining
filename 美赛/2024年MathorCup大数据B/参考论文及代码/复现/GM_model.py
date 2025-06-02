import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 实现灰色预测
def GM11(x0, k):  # （GM(1,1)）
    """
    :param x0: 初始数据
    :param k: 预测值个数
    """
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[:-1] + x1[1:])
    # 2. 构造数据矩阵和Y向量
    B = np.vstack([-z1, np.ones(len(x1) - 1)]).T
    Y = x0[1:]

    # 3. 使用最小二乘法估计参数a和b
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]

    # 4. 预测公式 x(t) = (x0(1) - b/a) * exp(-a * (t-1)) + b/a
    def predict(t):
        return (x0[0] - b / a) * np.exp(-a * (t - 1)) + b / a

    # 5. 计算累加生成序列的预测值
    x1_pred = [predict(i) for i in range(1, len(x0) + 1 + k)]

    # 6. 将累加生成序列的预测值转换为原始序列的预测值
    x0_pred = np.diff(x1_pred, prepend=0)

    return x0_pred


def GM21(x0, k):
    """
    :param x0: 初始数据
    :param k: 预测值个数
    """
    l = len(x0)
    x1 = np.cumsum(x0)
    a_x0 = np.diff(x0)
    z1 = 0.5 * (x1[:-1] + x1[1:])

    # 2. 构造数据矩阵B和向量Y
    B = np.vstack([-x0[1:], -z1, np.ones(len(x1) - 1)]).T
    Y = a_x0[1:]

    # 3. 使用最小二乘法估计参数a和b
    a1, a2, b = np.linalg.lstsq(B, Y, rcond=None)[0]

    # 4. 预测公式 x(k+1) = a * x(k) + b
    def predict(k):
        if k == 0:
            return x0[0]
        else:
            return a * predict(k - 1) + b

    # 5. 计算预测值
    x0_pred = [predict(i) for i in range(l + k)]

    return np.array(x0_pred)


def DGM21(x0, k):  # 离散灰色预测
    """
    :param x0: 初始数据
    :param k: 预测值个数
    """
    l = len(x0)
    x1 = np.cumsum(x0)
    a_x0 = np.diff(x0)
    z1 = 0.5 * (x1[:-1] + x1[1:])
    # print(a_x0)
    # 2. 构造数据矩阵B和向量Y
    B = np.vstack([-x0[1:], np.ones(len(x0) - 1)]).T
    Y = a_x0[:]
    # print(B, Y)
    # 3. 使用最小二乘法估计参数a和b
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]
    # print(a, b)

    # 4. 预测公式 x(k+1) =
    def predict(t):
        result = (-x0[0] / a + b / (a ** 2)) * np.exp(-a * (t - 1)) + b / a * (t - 1) + (1 + a) / a * x0[0] - b / (a ** 2)
        return result

    # 5. 计算预测值
    x1_pred = [predict(i) for i in range(2, l + k + 1)]
    x0_pred = np.diff(x1_pred, prepend=x0[0])    # 差分  x0[k] = x1[k] - x1[k - 1]
    x0_pred = np.insert(x0_pred, 0, x0[0])

    return x0_pred
