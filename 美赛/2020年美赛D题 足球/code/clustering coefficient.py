import numpy as np
import pandas as pd
import csv
def read_csv_to_matrix(file_path,n):
    # 初始化一个nxn的零矩阵
    W = np.zeros((n, n))
    maxweight = 0
    # 读取CSV文件
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            source = int(row['Source']) - 1  
            target = int(row['Target']) - 1  
            weight = int(row['Weight'])
            W[source, target] = weight

    W = W + W.T  # 无向图

    for i in range(n):
        for j in range(n):
            if W[i,j] > maxweight:
                maxweight = W[i,j]

    #W = W / maxweight   # 归一化

    return W, maxweight


def calculate_weighted_clustering_coefficient1(W):
    n = W.shape[0]  # 球员数量
    C_i = np.zeros(n)  # 初始化加权聚类系数矩阵

    # 计算邻接矩阵A
    A = (W > 0).astype(int)

    for i in range(n):
        k_i = np.sum(A[i, :])  # 节点i的度

        if k_i < 2:
            C_i[i] = 0  # 如果度小于2，则聚类系数为0
            continue
        # 计算<wi>
        avg_w_i = np.sum(W[i, :]) / k_i

        numerator = 0.0
        for j in range(n):
            for k in range(n):
                if j != i and k != i and j != k:
                    numerator += (W[i, j] + W[i, k]) / (2 * avg_w_i) * A[i, j] * A[i, k] * A[j, k]

        C_i[i] = numerator / (k_i * (k_i - 1))

    return C_i

def calculate_weighted_clustering_coefficient2(W):
    n = W.shape[0]  # 球员数量
    Cw_i = np.zeros(n)  # 初始化加权聚类系数矩阵

    for i in range(n):
        k_i = np.sum(W[i, :] > 0)  # 节点i的度

        if k_i < 2:
            Cw_i[i] = 0  # 如果度小于2，则聚类系数为0
            continue

        numerator = 0.0
        for j in range(n):
            for k in range(n):
                if j != i and k != i and j != k:
                    numerator += (W[i, j] * W[i, k] * W[j, k]) ** (1/3)

        Cw_i[i] = numerator / (k_i * (k_i - 1))

    return Cw_i

def calculate_weighted_clustering_coefficient3(W):
    n = W.shape[0]  # 球员数量
    C_W = np.zeros(n)  

    for i in range(n):
        numerator = 0.0
        denominator = 0.0
        cnt = 0
        for j in range(n):
            for k in range(n):
                if j != i and k != i and j != k:
                    
                    numerator += W[i, j] * W[j, k] * W[i, k]
                    denominator += W[i, j] * W[i, k]
                    #if W[i, j] != 0 and W[i, k] != 0:  cnt += 1

        if denominator != 0:
            C_W[i] = numerator / denominator
        else:
            C_W[i] = 0  

    return C_W



if __name__ == "__main__": 
    file_path = 'output_OOO9.csv'

    W, maxweight = read_csv_to_matrix(file_path,18)
    print("传球矩阵 W:")
    print(W)
    #C_w1 = calculate_weighted_clustering_coefficient1(W)
    #C_w2 = calculate_weighted_clustering_coefficient2(W/maxweight)
    C_w3 = calculate_weighted_clustering_coefficient3(W/maxweight)
    print("加权聚类系数矩阵 C_w:")
    print(C_w3)
    print(np.sum(C_w3)/18)
    #print(C_W3.shape)





