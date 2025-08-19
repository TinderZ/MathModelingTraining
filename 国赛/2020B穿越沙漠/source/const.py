import numpy as np
from collections import defaultdict
# 定义常量
WEATHER_LIST = [
    '晴朗', '高温', '晴朗', '晴朗', '晴朗', '晴朗', '高温', '高温', '高温','高温'
]

NUM_PLAYERS = 2
# 对手路径 (day 0 to 9)
OPPONENT_PATH_1 = [1, 4, 4, 6, 13, 13, 13, 13, 13, 13]
OPPONENT_PATH_2 = [1, 5, 5, 6, 13, 13, 13, 13, 13, 13]
OPPONENT_PATHS = [OPPONENT_PATH_1, OPPONENT_PATH_2]
OPPONENT_PATH_PROB = [0.5, 0.5]


MINE_NODES = [55]
VILLAGE_NODES = []
DESTINATION_NODES = [13]

END_NODE = 13
MAX_DAY = 10
INITIAL_MONEY = 10000
WEIGHT_LIMIT = 1200
BASE_INCOME = 200

# 资源参数
WATER_WEIGHT = 3
FOOD_WEIGHT = 2
WATER_PRICE = 5
FOOD_PRICE = 10
WATER_REFUND = 2.5
FOOD_REFUND = 5

# 基础消耗量 (晴朗, 高温, 沙暴)
BASE_CONSUMPTION = {
    '晴朗': (3, 4),
    '高温': (9, 9),
    '沙暴': (10, 10)
}

# 地图邻接表
GRAPH = {
    1: [2, 4, 5],
    2: [3, 4, 1],
    3: [2, 4, 8, 9],
    4: [1, 2, 3, 5, 6, 7],
    5: [1, 4, 6],
    6: [4, 5, 7, 12, 13],
    7: [4, 6, 11, 12],
    8: [3, 9],
    9: [3, 8, 10, 11],
    10: [9, 11, 13],
    11: [7, 9, 10, 12, 13],
    12: [7, 11, 13]
}

# 运动惯性路径定义
MOTION_INERTIA_PATHS = {
    # # 路径1: 23-21-9-15-13-12
    # (23, 21): 9,
    # (21, 9): 15,
    # (9, 15): 13,
    # (15, 13): 12,
    # # 路径2: 12-13-15 (反向)
    # (12, 13): 15,
    # (13, 15): 9,  # 15可以选择去9或13
    # # 路径3: 15-9-21-27
    # (15, 9): 21,
    # (9, 21): 27
}

# 各节点到终点的最短天数
SHORTEST_DAYS_TO_END = {
    1: 3,
    2: 3,
    3: 3,
    4: 2,
    5: 2,
    6: 1,
    7: 2,
    8: 3,
    9: 2,
    10: 1,
    11: 1,
    12: 1
}