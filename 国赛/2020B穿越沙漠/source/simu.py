from funcs import *
from const import *

def get_opponent_position_probability(day):
    """
    获取指定天数另一个玩家在各个位置的概率分布
    """
    prob_dist = {}
    for i, path in enumerate(OPPONENT_PATHS):
        if day < len(path):
            pos = path[day]
        else:
            # 超出路径长度后，停留在最后一个位置
            pos = path[-1]
        
        if pos in prob_dist:
            prob_dist[pos] += OPPONENT_PATH_PROB[i]
        else:
            prob_dist[pos] = OPPONENT_PATH_PROB[i]
    
    return prob_dist

def calculate_expected_consumption(base_water, base_food, pos, opponent_prob_dist, action_type):
    """
    计算考虑另一个玩家影响后的期望消耗
    action_type: 'move', 'mine'
    """
    # 获取当前位置另一个玩家存在的概率
    opponent_prob_at_pos = opponent_prob_dist.get(pos, 0)
         
    if action_type == 'move':
        # 移动：2倍基础消耗
        expected_water = 2 * base_water
        expected_food = 2 * base_food
        # 如果另一个玩家在同一区域移动，消耗可能受影响
        if opponent_prob_at_pos > 0:
            expected_water = 2 * expected_water * opponent_prob_at_pos + expected_water * (1 - opponent_prob_at_pos)
            expected_food = 2 * expected_food * opponent_prob_at_pos + expected_food * (1 - opponent_prob_at_pos)
            
    elif action_type == 'mine':
        # 挖矿：额外2倍消耗
        expected_water = base_water + 2 * base_water  # 基础 + 额外
        expected_food = base_food + 2 * base_food
        # 如果另一个玩家在同一矿山，收益可能受影响
        expected_income = BASE_INCOME
        if opponent_prob_at_pos > 0:
            # 根据题目描述，同一矿山时收益为基础的1/k倍
            # 这里假设k=2（两个玩家）
            expected_income = BASE_INCOME / NUM_PLAYERS
        return expected_water, expected_food, expected_income
    
    return expected_water, expected_food

def simulate_two_players():
    """
    双人游戏模拟函数
    """
    dp = initialize_dp()
    global_max_money = -1
    best_strategy = []
    
    for day in range(MAX_DAY):
        print(f"第{day+1}天")
        
        weather = WEATHER_LIST[day]
        base_water, base_food = get_base_consumption(weather)
        
        # 获取当天另一个玩家的位置概率分布
        opponent_prob_dist = get_opponent_position_probability(day)
        #print(f"对手位置概率分布: {opponent_prob_dist}")
        
        dp_next = dict()
        
        for state, (money, pre_pos, pre_money, pre_water, pre_food) in dp[day].items():
            pos, water, food = state
            if pos in DESTINATION_NODES: 
                continue

            # 剪枝：如果剩余天数不足以到达终点
            if (day) + SHORTEST_DAYS_TO_END.get(pos, float('inf')) > MAX_DAY:
                continue
            
            
            new_pos = pos
            new_water = water - expected_water
            new_food = food - expected_food
            new_money = money
            update_state(dp_next, (new_pos, new_water, new_food), new_money, pos, money, water, food)

            # 挖矿选项（如果在矿山）
            if pos in MINE_NODES:
                mine_water, mine_food, mine_income = calculate_expected_consumption(
                    base_water, base_food, pos, opponent_prob_dist, 'mine'
                )
                
                mine_water_final = water - mine_water
                mine_food_final = food - mine_food
                mine_money_final = money + mine_income
                update_state(dp_next, (new_pos, mine_water_final, mine_food_final), 
                           mine_money_final, pos, money, water, food)
            
            # 选项2: 行走到相邻节点
            for neighbor in GRAPH[pos]:
                
                # 计算移动的期望消耗
                expected_water, expected_food = calculate_expected_consumption(
                    base_water, base_food, pos, opponent_prob_dist, 'move'
                )
                
                new_pos = neighbor
                new_water = water - expected_water
                new_food = food - expected_food
                new_money = money
                
                # 检查是否到达终点
                if new_pos == END_NODE and new_water >= 0 and new_food >= 0 and new_money >= 0:
                    final_money = new_money + WATER_REFUND * new_water + FOOD_REFUND * new_food
                    if final_money > global_max_money:
                        global_max_money = final_money 
                        print(f"新的最大期望收益: {global_max_money}")
                        print(f"在第{day+1}天到达终点")
                        print(f"最终状态: 位置={new_pos}, 资金={new_money}, 水={new_water}, 食物={new_food}")
                    update_state(dp_next, (new_pos, new_water, new_food), new_money, pos, money, water, food)
                else:
                    update_state(dp_next, (new_pos, new_water, new_food), new_money, pos, money, water, food)

        dp[day + 1] = dp_next
    
    return global_max_money, dp

def trace_optimal_strategy(dp, global_max_money):
    """
    回溯最优策略路径
    """
    # 找到最优终点状态
    optimal_path = []
    
    # 从最后一天开始回溯
    for day in range(MAX_DAY, -1, -1):
        for state, (money, pre_pos, pre_money, pre_water, pre_food) in dp[day].items():
            pos, water, food = state
            if pos == END_NODE:
                final_money = money + WATER_REFUND * water + FOOD_REFUND * food
                if abs(final_money - global_max_money) < 1e-6:  # 找到最优解
                    optimal_path.append((day, pos, water, food, money))
                    # 继续回溯
                    current_state = state
                    current_day = day
                    while current_day > 0 and current_state in dp[current_day]:
                        _, pre_pos, pre_money, pre_water, pre_food = dp[current_day][current_state]
                        if pre_pos is not None:
                            current_day -= 1
                            current_state = (pre_pos, pre_water, pre_food)
                            optimal_path.append((current_day, pre_pos, pre_water, pre_food, pre_money))
                        else:
                            break
                    optimal_path.reverse()
                    return optimal_path
    
    return optimal_path

# 原有的单人游戏函数保持不变
def simulate():
    dp = initialize_dp()
    global_max_money = -1
    
    for day in range(MAX_DAY):
        print(f"第{day+1}天")

        weather = WEATHER_LIST[day]
        base_water, base_food = get_base_consumption(weather)
        dp_next = dict()
        
        for state, (money, pre_pos, pre_money, pre_water, pre_food) in dp[day].items():
            pos, water, food = state
            if pos in DESTINATION_NODES: continue

            # 剪枝：如果剩余天数不足以到达终点
            if (day) + SHORTEST_DAYS_TO_END.get(pos, float('inf')) > MAX_DAY:
                continue
            
            # 选项1: 停留当前位置
            new_pos = pos
            consume_water = base_water
            consume_food = base_food
            new_water = water - consume_water
            new_food = food - consume_food
            new_money = money
            update_state(dp_next, (new_pos, new_water, new_food), new_money, pos, money, water, food)

            if pos in MINE_NODES: # 挖矿# 停留矿山挖矿选项
                extra_water = 2 * base_water
                extra_food = 2 * base_food
                mine_water = new_water - extra_water
                mine_food = new_food - extra_food
                mine_money = new_money + BASE_INCOME
                update_state(dp_next, (new_pos, mine_water, mine_food), mine_money, pos, money, water, food)
            
            # 选项2: 行走到相邻节点
            for neighbor in GRAPH[pos]:
                # 运动惯性剪枝：如果有前一天位置，检查是否符合运动惯性
                if pre_pos is not None:
                    # 检查是否在运动惯性路径上
                    if (pre_pos, pos) in MOTION_INERTIA_PATHS:
                        expected_next = MOTION_INERTIA_PATHS[(pre_pos, pos)]
                        # 如果有明确的下一个位置要求，且当前选择不符合，则跳过
                        if expected_next is not None and neighbor != expected_next:
                            continue
                
                new_pos = neighbor
                consume_water = 2 * base_water
                consume_food = 2 * base_food
                new_water = water - consume_water
                new_food = food - consume_food
                new_money = money
                
                if new_pos == END_NODE and new_water >= 0 and new_food >= 0 and new_money >= 0:
                    final_money = new_money + WATER_REFUND * new_water + FOOD_REFUND * new_food
                    if final_money > global_max_money:
                        global_max_money = final_money 
                        print(f"新的最大收益: {global_max_money}")
                        print(f"新的最大收益在第{day+1}天结束")
                        print(f"新的最大收益状态: {new_pos, new_money, new_water, new_food}")
                        print(f"新的最大收益前一个状态: {pos, money, water, food}")
                    update_state(dp_next, (new_pos, new_water, new_food), new_money, pos, money, water, food)

                else:
                    update_state(dp_next, (new_pos, new_water, new_food), new_money, pos, money, water, food)

        dp[day + 1] = dp_next
        
    return global_max_money, dp

# 主函数示例
if __name__ == "__main__":
    print("=== 双人游戏模拟 ===")
    max_money, dp = simulate_two_players()
    print(f"\n最大期望收益: {max_money}")
    
    # 获取最优策略
    optimal_strategy = trace_optimal_strategy(dp, max_money)
    print("\n最优策略路径:")
    for step in optimal_strategy:
        day, pos, water, food, money = step
        print(f"第{day}天: 位置={pos}, 水={water}, 食物={food}, 资金={money}")