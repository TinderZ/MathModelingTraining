from const import *


def initialize_dp():
    dp = [dict() for _ in range(MAX_DAY + 1)]
    # 第0天：在起点购买资源
    for food in range(0, 180):
        for water in range(0, 195):
            # if food < water: continue
            weight = WATER_WEIGHT * water + FOOD_WEIGHT * food
            
            # 无法再多带一箱水或食物
            if (weight > WEIGHT_LIMIT):
                continue
            
            cost = WATER_PRICE * water + FOOD_PRICE * food
            if cost <= INITIAL_MONEY:
                money = INITIAL_MONEY - cost
                state = (1, water, food)
                dp[0][state] = (money, None, None, None, None)
    return dp
    
def get_base_consumption(weather):
    return BASE_CONSUMPTION[weather]

def update_state(dp_next, state, money, pre_pos, pre_money, pre_water, pre_food):
    pos, water, food = state
    # 检查资源非负
    if water < 0 or food < 0:
        return
    # 检查负重限制
    weight = WATER_WEIGHT * water + FOOD_WEIGHT * food
    if weight > WEIGHT_LIMIT:
        return
    # 更新状态：保留资金最大的
    if state in dp_next:
        if money > dp_next[state][0]:
            dp_next[state] = (money, pre_pos, pre_money, pre_water, pre_food)
    else:
        dp_next[state] = (money, pre_pos, pre_money, pre_water, pre_food)



def calculate_next_move_need(day, count):
    """
    计算从指定日期开始，接下来指定次数移动所需的总物资（水和食物）。

    Args:
        day (int): 开始计算的日期 (1-indexed)。
        move_count (int): 移动次数。

    Returns:
        tuple: (总共需要的水, 总共需要的食物)。
               如果无法完成指定次数移动（例如，日期超出范围），则返回None。
    """
    total_water_needed = 0
    total_food_needed = 0
    moves_count = 0
    current_day_index = day + 1   # 调整为day之后一天

    while moves_count < count:
        if current_day_index >= len(WEATHER_LIST):
            # 如果日期超出天气列表范围，则无法完成3次移动
            return total_water_needed, total_food_needed

        weather = WEATHER_LIST[current_day_index]
        base_water, base_food = BASE_CONSUMPTION[weather]

        if weather == '沙暴':
            # 沙暴天气，停留，消耗基础物资
            total_water_needed += base_water
            total_food_needed += base_food
        else:
            # 非沙暴天气，移动，消耗双倍物资
            total_water_needed += base_water * 2
            total_food_needed += base_food * 2
            moves_count += 1
        
        current_day_index += 1
        
    return total_water_needed, total_food_needed