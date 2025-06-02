import pandas as pd
import os

# 定义文件的根路径
root_path = r'E:\数模\训练题\2020年美赛D题 足球\data\passingevents\\'

# 创建一个列表来存储结果
results = []

# 循环处理1到38号文件
for x in range(1, 39):
    file_path = os.path.join(root_path, f'{x}.xlsx')
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 筛选出TeamID为Huskies的行
        df_huskies = df[df['TeamID'] == 'Huskies']
        
        # 1. TeamID为Huskies时，第N列之和与第L列绝对值之和
        Huskies_sum_deltaY = df_huskies['abs delta Y'].sum()
        Huskies_sum_deltaX = df_huskies['delta X'].abs().sum()
        advance_ratio = Huskies_sum_deltaY / Huskies_sum_deltaX
        
        # 2. TeamID为Huskies时, 第H,I,J,K列的均值
        mean_husk_x1 = df_huskies['x1'].mean(skipna=True)
        mean_husk_y1 = df_huskies['y1'].mean(skipna=True)
        mean_husk_x2 = df_huskies['x2'].mean(skipna=True)
        mean_husk_y2 = df_huskies['y2'].mean(skipna=True)
        
        # 筛选出TeamID不为Huskies的行
        df_not_huskies = df[df['TeamID'] != 'Huskies']
        
        # 3. TeamID不为Huskies时，第N列之和与第L列绝对值之和
        sum_n_nh = df_not_huskies['abs delta Y'].sum()
        sum_abs_l_nh = df_not_huskies['delta X'].abs().sum()
        adv_ratio_unhusk = sum_n_nh / sum_abs_l_nh
        
        # 4. TeamID不为Huskies时, 第H,I,J,K列的均值
        mean_nh_x1 = df_not_huskies['x1'].mean(skipna=True)
        mean_nh_y1 = df_not_huskies['y1'].mean(skipna=True)
        mean_nh_x2 = df_not_huskies['x2'].mean(skipna=True)
        mean_nh_y2 = df_not_huskies['y2'].mean(skipna=True)
        
        # 将结果添加到results列表中
        results.append([x, Huskies_sum_deltaY, Huskies_sum_deltaX, advance_ratio, mean_husk_x1, mean_husk_y1, mean_husk_x2, mean_husk_y2, 
                        sum_n_nh, sum_abs_l_nh, adv_ratio_unhusk, mean_nh_x1, mean_nh_y1, mean_nh_x2, mean_nh_y2])
    
    except Exception as e:
        print(f"处理文件{x}.xlsx时出错: {e}")
        # 跳过这个文件，继续处理下一个
        continue

# 定义结果列名
columns = ['MatchID', 'Husk_sumdy', 'Husk_sumdx','advance_ratio','Mean_Husk_x1', 'Mean_Husk_y1', 'Mean_Husk_x2', 'Mean_Husk_y2', 
           'Oppoent_sumdy','Oppoent_sumdx','advance_ratio2', 'Mean_NH_x1', 'Mean_NH_y1', 'Mean_NH_x2', 'Mean_NH_y2']

# 将结果存储到DataFrame中
result_df = pd.DataFrame(results, columns=columns)

# 将结果写入Excel文件
output_path = os.path.join(root_path, 'advance ratio_eachmatch.xlsx')
result_df.to_excel(output_path, index=False)