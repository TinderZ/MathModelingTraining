import pandas as pd
import matplotlib.pyplot as plt


# 读取表格文件中的工作表
file_path = 'q1_timeseries.xlsx'

sheet1_data = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
# 将日期字段转换为日期格式
sheet1_data['Date'] = pd.to_datetime(sheet1_data['Date'])
# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(sheet1_data['Date'], sheet1_data['Number of  reported results'], marker='x')
plt.title('Daily attendee trends')
plt.xlabel('Date')
plt.ylabel('Number of results')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# 显示图表
plt.show()