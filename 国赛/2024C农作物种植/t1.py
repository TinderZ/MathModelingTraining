import gurobipy as gurobi
from gurobipy import GRB

def setup_planting_model_constraints():
    """
    建立农作物种植策略的数学规划模型并添加约束条件。
    目标函数部分未包含在此函数中。
    """
    # --- 1. 数据准备 ---

    # 地块数据 (根据图片手动整理)
    # 格式: (地块名称, 类型, 面积/亩)
    land_data = [
        ('A1', '平旱地', 80), ('A2', '平旱地', 55), ('A3', '平旱地', 35), ('A4', '平旱地', 72), ('A5', '平旱地', 68), ('A6', '平旱地', 55),
        ('B1', '梯田', 60), ('B2', '梯田', 46), ('B3', '梯田', 40), ('B4', '梯田', 28), ('B5', '梯田', 25), ('B6', '梯田', 86),
        ('B7', '梯田', 55), ('B8', '梯田', 44), ('B9', '梯田', 50), ('B10', '梯田', 25), ('B11', '梯田', 60), ('B12', '梯田', 45),
        ('B13', '梯田', 32), ('B14', '梯田', 20),
        ('C1', '山坡地', 15), ('C2', '山坡地', 13), ('C3', '山坡地', 15), ('C4', '山坡地', 18), ('C5', '山坡地', 27), ('C6', '山坡地', 20),
        ('D1', '水浇地', 15), ('D2', '水浇地', 10), ('D3', '水浇地', 14), ('D4', '水浇地', 6), ('D5', '水浇地', 10), ('D6', '水浇地', 12),
        ('D7', '水浇地', 22), ('D8', '水浇地', 20),
        ('E1', '普通大棚', 0.6), ('E2', '普通大棚', 0.6), ('E3', '普通大棚', 0.6), ('E4', '普通大棚', 0.6), ('E5', '普通大棚', 0.6),
        ('E6', '普通大棚', 0.6), ('E7', '普通大棚', 0.6), ('E8', '普通大棚', 0.6), ('E9', '普通大棚', 0.6), ('E10', '普通大棚', 0.6),
        ('E11', '普通大棚', 0.6), ('E12', '普通大棚', 0.6), ('E13', '普通大棚', 0.6), ('E14', '普通大棚', 0.6), ('E15', '普通大棚', 0.6),
        ('E16', '普通大棚', 0.6),
        ('F1', '智慧大棚', 0.6), ('F2', '智慧大棚', 0.6), ('F3', '智慧大棚', 0.6), ('F4', '智慧大棚', 0.6)
    ]

    # 模型参数
    num_lands = 54
    num_crops = 41
    num_years = 7  # 2024-2030
    seasons_per_year = 2
    num_seasons = num_years * seasons_per_year # 总季度数

    # 索引集合
    I_lands = range(1, num_lands + 1)
    J_crops = range(1, num_crops + 1)
    T_seasons = range(1, num_seasons + 1)

    # 地块面积 Fi
    F = {i: data[2] for i, data in zip(I_lands, land_data)}

    # 定义地块类型集合 (根据问题描述 C题.pdf)
    # 平旱地、梯田、山坡地每年只能种植一季 
    I_yearly_crop = range(1, 27) # 地块 A, B, C
    # 水浇地 
    I_irrigated = range(27, 35) # 地块 D
    # 普通大棚 
    I_normal_greenhouse = range(35, 51) # 地块 E
    # 智慧大棚 
    I_smart_greenhouse = range(51, 55) # 地块 F

    # 作物数据 (根据附件2整理)
    # 格式: (作物编号, 作物名称, 地块类型, 种类, 种植成本(元/亩), 亩产量(斤), 销售单价平均值(元/斤))
    crop_data = [
        (1, '黄豆', '平旱地', '单季', 500, 400, 6.75),
        (2, '黑豆', '平旱地', '单季', 400, 350, 7.50),
        (3, '红豆', '平旱地', '单季', 350, 350, 6.50),
        (4, '绿豆', '平旱地', '单季', 415, 350, 7.25),
        (5, '爬豆', '平旱地', '单季', 800, 450, 3.50),
        (6, '小麦', '平旱地', '单季', 300, 500, 2.75),
        (7, '玉米', '平旱地', '单季', 400, 360, 6.25),
        (8, '谷子', '平旱地', '单季', 630, 400, 5.75),
        (9, '高粱', '平旱地', '单季', 525, 360, 6.75),
        (10, '黍子', '平旱地', '单季', 110, 350, 30.50),
        (11, '荞麦', '平旱地', '单季', 3000, 1000, 1.50),
        (12, '南瓜', '平旱地', '单季', 2300, 2000, 2.75),
        (13, '红薯', '平旱地', '单季', 420, 400, 5.25),
        (14, '莜麦', '平旱地', '单季', 525, 350, 3.50),
        (15, '大麦', '梯田', '单季', 380, 400, 2.75),
        (16, '水稻', '梯田', '单季', 475, 400, 6.75),
        (17, '红豆', '梯田', '单季', 380, 350, 7.50),
        (18, '玉米', '梯田', '单季', 330, 350, 6.50),
        (19, '豌豆', '梯田', '单季', 395, 350, 6.25),
        (20, '土豆', '梯田', '单季', 760, 450, 3.50),
        (21, '玉米', '梯田', '单季', 950, 500, 2.75),
        (22, '高粱', '梯田', '单季', 380, 360, 6.25),
        (23, '黍子', '梯田', '单季', 600, 400, 5.75),
        (24, '荞麦', '梯田', '单季', 500, 360, 6.75),
        (25, '燕麦', '梯田', '单季', 105, 350, 30.50),
        (26, '大麦', '山坡地', '单季', 2850, 1600, 1.50),
        (27, '谷子', '山坡地', '单季', 2100, 2000, 2.75),
        (28, '高粱', '山坡地', '单季', 100, 400, 5.25),
        (29, '玉米', '山坡地', '单季', 500, 350, 3.50),
        (30, '大麦', '山坡地', '单季', 360, 400, 2.75),
        (31, '红豆', '山坡地', '单季', 450, 350, 7.50),
        (32, '玉米', '山坡地', '单季', 360, 350, 6.50),
        (33, '豌豆', '山坡地', '单季', 315, 350, 6.25),
        (34, '土豆', '山坡地', '单季', 375, 350, 6.25),
        (35, '玉米', '山坡地', '单季', 720, 450, 3.50),
        (36, '高粱', '山坡地', '单季', 900, 500, 2.75),
        (37, '黍子', '山坡地', '单季', 360, 360, 6.25),
        (38, '荞麦', '山坡地', '单季', 570, 400, 5.75),
        (39, '燕麦', '山坡地', '单季', 475, 360, 6.75),
        (40, '大麦', '山坡地', '单季', 100, 350, 30.50),
        (41, '谷子', '山坡地', '单季', 2700, 1000, 1.50)
    ]
    
    # 种植成本 (元/亩)
    planting_cost = {i: crop_data[i-1][4] for i in range(1, num_crops + 1)}
    
    # 亩产量 (斤)
    yield_per_mu = {i: crop_data[i-1][5] for i in range(1, num_crops + 1)}
    
    # 销售单价 (元/斤) - 已计算平均值
    selling_price = {i: crop_data[i-1][6] for i in range(1, num_crops + 1)}
    
    # 根据实际作物名称重新定义作物类型集合
    # 豆类作物 (黄豆、黑豆、红豆、绿豆、爬豆、豌豆)
    J_bean = set([1, 2, 3, 4, 5, 19, 31, 33])
    
    # 粮食作物 (包括豆类、小麦、玉米、谷子、高粱、黍子、荞麦、莜麦、大麦、燕麦)
    J_grain = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41])
    
    # 水稻
    J_rice = set([16])
    
    # 蔬菜作物 (南瓜、红薯、土豆)
    J_vegetable = set([12, 13, 20, 34])
    
    # 食用菌 (暂时为空，根据实际情况可能需要从其他数据源补充)
    J_fungi = set()
    
    # --- 2. 模型初始化 ---
    model = gurobi.Model("Planting_Strategy_Constraints")

    # --- 3. 决策变量 ---
    # E_ijt: 0-1变量，如果地块i在t季种植作物j，则为1 [cite: 9]
    E = model.addVars(I_lands, J_crops, T_seasons, vtype=GRB.BINARY, name="E")
    
    # X_ijt: 连续变量，表示地块i在t季种植作物j的面积 [cite: 14]
    X = model.addVars(I_lands, J_crops, T_seasons, vtype=GRB.CONTINUOUS, lb=0, name="X")



    # --- 4. 约束条件 ---

    # **约束1: 种植决策与种植面积的关联**
    # 如果不种植某种作物 (E_ijt=0)，则其面积必须为0 (X_ijt=0)。
    # 如果种植 (E_ijt=1)，则面积必须大于等于一个下限 (例如地块面积的1/10)，且小于等于总面积。 
    # 这个约束也体现了“每种作物在单个地块种植的面积不宜太小”的要求。 
    model.addConstrs((X[i, j, t] <= F[i] * E[i, j, t] for i in I_lands for j in J_crops for t in T_seasons), name="C1a_area_upper_bound")
    # m.addConstrs((X[i, j, t] >= (F[i] / 10) * E[i, j, t] for i in I_lands for j in J_crops for t in T_seasons), name="C1b_area_lower_bound") # 可选的最小面积约束

    # **约束2: 总种植面积不超过地块面积**
    # 每个地块在任何一个季度的总种植面积都不能超过该地块的总面积。 
    model.addConstrs((gurobi.quicksum(X[i, j, t] for j in J_crops) <= F[i] for i in I_lands for t in T_seasons), name="C2_total_area")

    # **约束3: 作物不能连续重茬种植**
    # 同一地块的同种作物不能连续两年在同一个季节种植。 
    # E_ij^t + E_ij^(t+2) <= 1, 这等价于 E_ij^t * E_ij^(t+2) = 0 [cite: 19]
    model.addConstrs((E[i, j, t] + E[i, j, t + 2] <= 1 for i in I_lands for j in J_crops for t in T_seasons if t <= num_seasons - 2), name="C3_no_consecutive_planting")

    # **约束4: 三年内至少种植一次豆类作物**
    # 每个地块从2023年开始，三年内所有土地至少要种植一次豆类作物。 
    # 我们为2024-2026和2027-2029两个三年窗口设置此约束。
    # 第一个三年窗口 (2024-2026)，对应t=1到6
    model.addConstrs((gurobi.quicksum(E[i, j, t] for j in J_bean for t in range(1, 7)) >= 1 for i in I_lands), name="C4a_bean_2024_2026")
    # 第二个三年窗口 (2027-2029)，对应t=7到12
    model.addConstrs((gurobi.quicksum(E[i, j, t] for j in J_bean for t in range(7, 13)) >= 1 for i in I_lands), name="C4b_bean_2027_2029")

    # **约束5: 不同类型地块的种植规则**
    # 这些规则是根据 C题.pdf  的描述建立的，它比您提供的PDF中的约束描述更清晰。

    # 5a: 平旱地、梯田、山坡地 (每年一季粮食)
    for i in I_yearly_crop:
        # 只能种植粮食作物
        model.addConstrs((E[i, j, t] == 0 for j in J_crops if j not in J_grain for t in T_seasons), name=f"C5a_grain_only_land_{i}")
        # 每年只能种植一季 (假设在春季，即t为奇数)
        for t in T_seasons:
            if t % 2 == 0: # 偶数季度 (秋季) 不允许种植
                model.addConstr(gurobi.quicksum(E[i, j, t] for j in J_crops) == 0, name=f"C5a_one_season_land_{i}_t_{t}")

    # 5b: 水浇地 (一季水稻或两季蔬菜)
    for i in I_irrigated:
        # 只能种植水稻或蔬菜
        model.addConstrs((E[i, j, t] == 0 for j in J_crops if j not in J_rice and j not in J_vegetable for t in T_seasons), name=f"C5b_rice_veg_only_land_{i}")
        for year in range(num_years):
            t1 = year * 2 + 1 # 当年春季
            t2 = year * 2 + 2 # 当年秋季
            # 引入辅助变量来控制选择
            is_rice_year = model.addVar(vtype=GRB.BINARY, name=f"is_rice_year_{i}_{year+2024}")
            
            # 如果种植水稻，只能在春季，且秋季休耕
            model.addConstr(gurobi.quicksum(E[i, j, t1] for j in J_rice) <= is_rice_year, name=f"C5b_rice_decision_t1_{i}_{year+2024}")
            model.addConstr(gurobi.quicksum(E[i, j, t2] for j in J_crops) <= (1 - is_rice_year), name=f"C5b_fallow_if_rice_{i}_{year+2024}")

            # 如果不种水稻，则可以种两季蔬菜
            model.addConstr(gurobi.quicksum(E[i, j, t1] for j in J_vegetable) <= (1 - is_rice_year), name=f"C5b_veg_if_not_rice_t1_{i}_{year+2024}")
            model.addConstr(gurobi.quicksum(E[i, j, t2] for j in J_vegetable) <= (1 - is_rice_year), name=f"C5b_veg_if_not_rice_t2_{i}_{year+2024}")


    # 5c: 普通大棚 (一季蔬菜和一季食用菌)
    for i in I_normal_greenhouse:
        # 只能种植蔬菜和食用菌
        model.addConstrs((E[i, j, t] == 0 for j in J_crops if j not in J_vegetable and j not in J_fungi for t in T_seasons), name=f"C5c_veg_fungi_only_land_{i}")
        for year in range(num_years):
            t1 = year * 2 + 1
            t2 = year * 2 + 2
            # 每年必须种植一季蔬菜
            model.addConstr(gurobi.quicksum(E[i, j, t] for j in J_vegetable for t in [t1, t2]) == 1, name=f"C5c_one_veg_per_year_{i}_{year+2024}")
            # 每年必须种植一季食用菌
            model.addConstr(gurobi.quicksum(E[i, j, t] for j in J_fungi for t in [t1, t2]) == 1, name=f"C5c_one_fungi_per_year_{i}_{year+2024}")

    # 5d: 智慧大棚 (两季蔬菜)
    for i in I_smart_greenhouse:
        # 只能种植蔬菜
        model.addConstrs((E[i, j, t] == 0 for j in J_crops if j not in J_vegetable for t in T_seasons), name=f"C5d_veg_only_land_{i}")
        for year in range(num_years):
            t1 = year * 2 + 1
            t2 = year * 2 + 2
            # 每季都必须种植蔬菜
            model.addConstr(gurobi.quicksum(E[i, j, t1] for j in J_vegetable) == 1, name=f"C5d_veg_t1_{i}_{year+2024}")
            model.addConstr(gurobi.quicksum(E[i, j, t2] for j in J_vegetable) == 1, name=f"C5d_veg_t2_{i}_{year+2024}")
    
    model.update()
    print("所有约束条件添加完毕。")
    print(f"模型包含 {model.numVars} 个变量和 {model.numConstrs} 个约束。")
    
    return model

# --- 执行函数 ---
if __name__ == '__main__':
    # 调用函数来构建模型和约束
    model = setup_planting_model_constraints()
    
    # 此时，模型 `model` 已经包含了所有的决策变量和约束条件。
    # 接下来您可以在这里添加目标函数 (Objective Function) 并进行求解。
    # 例如:
    # model.setObjective(..., GRB.MAXIMIZE)
    # model.optimize()
    
    print("\n模型构建成功。您可以继续添加目标函数并求解。")