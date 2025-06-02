import pandas as pd
import numpy as np
from collections import deque
import os

def process_half_data(df_half, half_label):
    # 按EventTime排序
    df_half = df_half.sort_values(by='EventTime')
    
    # 找到第50次传球的时间t0
    if len(df_half) < 50:
        return pd.DataFrame()  # 数据不足50次传球，返回空数据框
    
    t0 = df_half.iloc[49]['EventTime']
    
    # 初始化滑动窗口
    window = deque(maxlen=50)
    
    results = []
    # 遍历数据
    for index, row in df_half.iterrows():
        if row['EventTime'] >= t0:
            window.append(row)
            if len(window) == 50:
                # 计算平均x坐标
                x1_values = [w['x1'] for w in window]
                x2_values = [w['x2'] for w in window]
                avg_x = (sum(x1_values) + sum(x2_values)) / (2 * 50)
                
                # 计算平均y坐标
                y1_values = [w['y1'] for w in window]
                y2_values = [w['y2'] for w in window]
                avg_y = (sum(y1_values) + sum(y2_values)) / (2 * 50)
                
                # 计算推进率
                delta_y = sum(abs(w['y2'] - w['y1']) for w in window)
                delta_x = sum(abs(w['x2'] - w['x1']) for w in window)
                advancement_rate = delta_y / delta_x if delta_x != 0 else 0
                
                # 质心坐标
                centroid_x = avg_x
                centroid_y = avg_y
                
                # 计算Centrality dispersion
                points = [(w['x1'], w['y1']) for w in window] + [(w['x2'], w['y2']) for w in window]
                distances = [np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2) for x, y in points]
                std_dev = np.std(distances)
                
                # 记录时间t，avg_x, avg_y, 推进率, std_dev
                t = window[-1]['EventTime']  # 窗口最后一次传球的时间
                results.append([half_label, t, avg_x, avg_y, advancement_rate, std_dev])
    
    # 创建结果数据框
    if results:
        result_df = pd.DataFrame(results, columns=['Half', 'Time', 'AvgX', 'AvgY', 'AdvancementRate', 'CentralityDispersion'])
    else:
        result_df = pd.DataFrame(columns=['Half', 'Time', 'AvgX', 'AvgY', 'AdvancementRate', 'CentralityDispersion'])
    
    return result_df









if __name__ == '__main__':
    base_dir = "E:/数模/训练题/2020年美赛D题 足球/data/passingevents"
    
    for x in range(1, 39):
        output_dir = f"{base_dir}/50_passing"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 处理Huskies队的数据
        huskies_file = f"{base_dir}/_eachmatch_/{x}_H_edge.xlsx"
        df_huskies = pd.read_excel(huskies_file, sheet_name='Sheet1')
        
        # 分离第一半场和第二半场数据
        df_1H = df_huskies[df_huskies['MatchPeriod'] == '1H']
        df_2H = df_huskies[df_huskies['MatchPeriod'] == '2H']
        
        # 处理第一半场
        result_1H = process_half_data(df_1H, '1H')
        
        # 处理第二半场
        result_2H = process_half_data(df_2H, '2H')
        
        # 合并结果，并插入空白行
        if not result_1H.empty and not result_2H.empty:
            combined_results = pd.concat([result_1H, pd.DataFrame([['', '', '', '', '', '']], columns=result_1H.columns), result_2H], ignore_index=True)
        elif not result_1H.empty:
            combined_results = result_1H
        elif not result_2H.empty:
            combined_results = result_2H
        else:
            combined_results = pd.DataFrame(columns=['Half', 'Time', 'AvgX', 'AvgY', 'AdvancementRate', 'CentralityDispersion'])
        
        # 保存结果
        huskies_output = f"{base_dir}/50_passing/{x}_H_50passing.xlsx"
        combined_results.to_excel(huskies_output, index=False)
        



        
        # 处理对手的数据
        opponent_file = f"{base_dir}/_eachmatch_/{x}_O_edge.xlsx"
        df_opponent = pd.read_excel(opponent_file, sheet_name='Sheet1')
        
        # 分离第一半场和第二半场数据
        df_1H_opponent = df_opponent[df_opponent['MatchPeriod'] == '1H']
        df_2H_opponent = df_opponent[df_opponent['MatchPeriod'] == '2H']
        
        # 处理第一半场
        result_1H_opponent = process_half_data(df_1H_opponent, '1H')
        
        # 处理第二半场
        result_2H_opponent = process_half_data(df_2H_opponent, '2H')
        
        # 合并结果，并插入空白行
        if not result_1H_opponent.empty and not result_2H_opponent.empty:
            combined_results_opponent = pd.concat([result_1H_opponent, pd.DataFrame([['', '', '', '', '', '']], columns=result_1H_opponent.columns), result_2H_opponent], ignore_index=True)
        elif not result_1H_opponent.empty:
            combined_results_opponent = result_1H_opponent
        elif not result_2H_opponent.empty:
            combined_results_opponent = result_2H_opponent
        else:
            combined_results_opponent = pd.DataFrame(columns=['Half', 'Time', 'AvgX', 'AvgY', 'AdvancementRate', 'CentralityDispersion'])
        
        # 保存结果
        opponent_output = f"{base_dir}/50_passing/{x}_O_50passing.xlsx"
        combined_results_opponent.to_excel(opponent_output, index=False)