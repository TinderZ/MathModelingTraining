import pandas as pd
import numpy as np
from pathlib import Path

def extract_paper_format_data():
    """
    从原始Excel文件中提取论文格式所需的数据
    提取特定时间点的位置和速度数据
    """
    # 原始文件路径
    input_file = Path(r"e:\MathematicalModeling\MathematicalModelingTraining\国赛\2024A板凳龙\result1.xlsx")
    
    # 输出文件路径
    output_file = Path(r"e:\MathematicalModeling\MathematicalModelingTraining\国赛\2024A板凳龙\paper_format_data.xlsx")
    
    try:
        # 读取原始数据
        print("正在读取原始Excel文件...")
        df_pos = pd.read_excel(input_file, index_col=0, sheet_name="位置")
        df_vel = pd.read_excel(input_file, index_col=0, sheet_name="速度")
        
        # 需要提取的时间点（秒）
        time_points = [0, 60, 120, 180, 240, 300]
        time_columns = [f"{t} s" for t in time_points]
        
        # 检查时间列是否存在
        missing_columns = [col for col in time_columns if col not in df_pos.columns]
        if missing_columns:
            print(f"警告：以下时间列在原始数据中不存在：{missing_columns}")
            # 只使用存在的列
            time_columns = [col for col in time_columns if col in df_pos.columns]
        
        # 提取位置数据
        print("正在提取位置数据...")
        
        # 定义需要的行（根据图片中的表格格式）
        position_rows = [
            "龙头x (m)", "龙头y (m)",
            "第1节龙身x (m)", "第1节龙身y (m)",
            "第51节龙身x (m)", "第51节龙身y (m)", 
            "第101节龙身x (m)", "第101节龙身y (m)",
            "第151节龙身x (m)", "第151节龙身y (m)",
            "第201节龙身x (m)", "第201节龙身y (m)",
            "龙尾（后）x (m)", "龙尾（后）y (m)"
        ]
        
        # 检查行是否存在
        available_position_rows = [row for row in position_rows if row in df_pos.index]
        missing_position_rows = [row for row in position_rows if row not in df_pos.index]
        
        if missing_position_rows:
            print(f"警告：以下位置行在原始数据中不存在：{missing_position_rows}")
        
        # 直接提取位置数据
        position_data = df_pos.loc[available_position_rows, time_columns]
        
        print("位置数据提取完成")
        
        # 现在处理速度数据
        print("正在处理速度数据...")
        
        # 定义需要的速度行（直接使用原文件中的名称）
        velocity_rows = [
            "龙头 (m/s)",
            "第1节龙身  (m/s)",
            "第51节龙身  (m/s)",
            "第101节龙身  (m/s)", 
            "第151节龙身  (m/s)",
            "第201节龙身  (m/s)",
            "龙尾（后） (m/s)"
        ]
        
        # 检查速度行是否存在于原始文件中
        available_velocity_rows = [row for row in velocity_rows if row in df_vel.index]
        
        if available_velocity_rows:
            print("在原始文件中找到速度数据")
            velocity_data = df_vel.loc[available_velocity_rows, time_columns]
        
        
        print("速度数据处理完成")
        
        # 保存到Excel文件
        print("正在保存数据到Excel文件...")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            position_data.to_excel(writer, sheet_name='表1_论文中位置结果的格式')
            velocity_data.to_excel(writer, sheet_name='表2_论文中速度结果的格式')
        
        print(f"数据提取完成！结果已保存到：{output_file}")
        
        # 显示提取的数据预览
        print("\n=== 位置数据预览 ===")
        print(position_data)
        
        print("\n=== 速度数据预览 ===")
        print(velocity_data)
        
        return position_data, velocity_data
        
    except Exception as e:
        print(f"处理过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    extract_paper_format_data()