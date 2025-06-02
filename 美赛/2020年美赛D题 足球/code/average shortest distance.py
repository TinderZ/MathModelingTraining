import numpy as np
import heapq
import csv


def read_csv_to_matrix(file_path):
    # 初始化一个11x11的零矩阵
    W = np.zeros((18, 18))

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            source = int(row['Source']) - 1  
            target = int(row['Target']) - 1  
            weight = int(row['Weight'])
            W[source, target] = weight
    
    W = (W + W.T)   # 无向图
    
    return W


def dijkstra(W, start):
    n = W.shape[0]
    distances = {i: float('inf') for i in range(n)}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue


        for neighbor in range(n):
            if W[current_node, neighbor] > 0:
                distance = current_distance + (1 / W[current_node, neighbor])

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def calculate_average_shortest_path(W):
    n = W.shape[0]
    total_shortest_paths = 0

    for i in range(n):
        distances = dijkstra(W, i)
        print(distances)
        for j in range(n):
            if i != j: #and distances[j] != float('inf'):
                total_shortest_paths += distances[j]

    average_shortest_path = total_shortest_paths / (n * (n - 1))
    return average_shortest_path

# 示例使用
if __name__ == "__main__":
    file_path = "output_OOO9.csv"
    W = read_csv_to_matrix(file_path)
    print("传球矩阵 W:")
    print(W)

    average_shortest_path = calculate_average_shortest_path(W)
    print("平均最短路径:", average_shortest_path)