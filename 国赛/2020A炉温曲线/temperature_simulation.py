import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

class ReflowOvenSimulator:
    def __init__(self):
        # 炉子参数
        self.zone_length = 30.5  # cm，小温区长度
        self.gap_length = 5      # cm，间隙长度
        self.front_length = 25   # cm，炉前区域长度
        self.back_length = 25    # cm，炉后区域长度
        self.ambient_temp = 25   # °C，环境温度
        self.thickness = 0.15    # mm，焊接区域厚度
        
        # 物理参数（需要通过实验数据拟合）
        self.heat_transfer_coeff = None  # 传热系数
        self.thermal_capacity = None     # 热容
        
    def load_experimental_data(self, file_path):
        """读取实验数据"""
        try:
            # 尝试读取Excel文件
            df = pd.read_excel(file_path)
            # 假设第一列是时间，第二列是温度
            time_col = df.columns[0]
            temp_col = df.columns[1]
            
            self.exp_time = df[time_col].values
            self.exp_temp = df[temp_col].values
            
            # 移除NaN值
            mask = ~(np.isnan(self.exp_time) | np.isnan(self.exp_temp))
            self.exp_time = self.exp_time[mask]
            self.exp_temp = self.exp_temp[mask]
            
            print(f"成功读取实验数据：{len(self.exp_time)}个数据点")
            return True
        except Exception as e:
            print(f"读取实验数据失败：{e}")
            return False
    
    def calculate_position(self, time, speed):
        """根据时间和速度计算位置"""
        return speed * time / 60  # 转换为cm/s
    
    def get_zone_temp(self, position, zone_temps):
        """根据位置获取对应温区的设定温度"""
        # 炉前区域
        if position < self.front_length:
            return self.ambient_temp
        
        # 计算在哪个温区
        pos_in_oven = position - self.front_length
        
        # 11个小温区
        for i in range(11):
            zone_start = i * (self.zone_length + self.gap_length)
            zone_end = zone_start + self.zone_length
            
            if zone_start <= pos_in_oven < zone_end:
                # 在温区内
                if i < 5:  # 小温区1-5
                    return zone_temps[0]
                elif i == 5:  # 小温区6
                    return zone_temps[1]
                elif i == 6:  # 小温区7
                    return zone_temps[2]
                elif i in [7, 8]:  # 小温区8-9
                    return zone_temps[3]
                else:  # 小温区10-11
                    return self.ambient_temp
            elif zone_end <= pos_in_oven < zone_end + self.gap_length:
                # 在间隙中，取相邻温区的平均值
                if i < 4:
                    return zone_temps[0]
                elif i == 4:
                    return (zone_temps[0] + zone_temps[1]) / 2
                elif i == 5:
                    return (zone_temps[1] + zone_temps[2]) / 2
                elif i == 6:
                    return (zone_temps[2] + zone_temps[3]) / 2
                elif i == 7:
                    return zone_temps[3]
                elif i == 8:
                    return (zone_temps[3] + self.ambient_temp) / 2
                else:
                    return self.ambient_temp
        
        # 炉后区域
        return self.ambient_temp
    
    def temperature_ode(self, T, t, speed, zone_temps):
        """温度变化的微分方程"""
        position = self.calculate_position(t, speed)
        T_zone = self.get_zone_temp(position, zone_temps)
        
        # 简化的传热模型：dT/dt = h * (T_zone - T) / (ρ * c * δ)
        # 这里使用拟合得到的参数
        if self.heat_transfer_coeff is None:
            # 默认参数，需要通过实验数据拟合
            h_eff = 0.1  # 有效传热系数
        else:
            h_eff = self.heat_transfer_coeff
        
        dTdt = h_eff * (T_zone - T)
        return dTdt
    
    def fit_parameters(self):
        """通过实验数据拟合模型参数"""
        if not hasattr(self, 'exp_time') or not hasattr(self, 'exp_temp'):
            print("请先加载实验数据")
            return False
        
        # 实验条件
        exp_speed = 70  # cm/min
        exp_zone_temps = [175, 195, 235, 255]  # °C
        
        def model_temp(t, h_eff):
            """模型温度计算函数"""
            T0 = 30  # 初始温度
            try:
                T = odeint(lambda T, t: h_eff * (self.get_zone_temp(
                    self.calculate_position(t, exp_speed), exp_zone_temps) - T), 
                    T0, t)
                return T.flatten()
            except:
                return np.full_like(t, T0)
        
        try:
            # 拟合传热系数
            popt, _ = curve_fit(model_temp, self.exp_time, self.exp_temp, 
                              bounds=(0.01, 1.0), maxfev=1000)
            self.heat_transfer_coeff = popt[0]
            print(f"拟合得到的传热系数：{self.heat_transfer_coeff:.4f}")
            return True
        except Exception as e:
            print(f"参数拟合失败：{e}")
            # 使用默认值
            self.heat_transfer_coeff = 0.1
            return False
    
    def simulate_temperature(self, speed, zone_temps, time_step=0.5, max_time=600):
        """仿真温度变化"""
        # 时间数组
        time_array = np.arange(0, max_time, time_step)
        
        # 初始温度
        T0 = 30  # °C
        
        # 求解微分方程
        try:
            temperature = odeint(self.temperature_ode, T0, time_array, 
                               args=(speed, zone_temps))
            temperature = temperature.flatten()
        except Exception as e:
            print(f"温度仿真失败：{e}")
            return None, None
        
        return time_array, temperature
    
    def get_specific_positions_temp(self, time_array, temperature, speed, zone_temps):
        """获取指定位置的温度"""
        results = {}
        
        # 计算各个关键位置
        positions = {
            '小温区3中点': self.front_length + 2.5 * (self.zone_length + self.gap_length) + self.zone_length/2,
            '小温区6中点': self.front_length + 5.5 * (self.zone_length + self.gap_length) + self.zone_length/2,
            '小温区7中点': self.front_length + 6.5 * (self.zone_length + self.gap_length) + self.zone_length/2,
            '小温区8结束': self.front_length + 8 * (self.zone_length + self.gap_length)
        }
        
        for pos_name, pos_cm in positions.items():
            # 找到对应的时间
            target_time = pos_cm * 60 / speed  # 转换为秒
            
            # 在时间数组中找到最接近的点
            idx = np.argmin(np.abs(time_array - target_time))
            
            results[pos_name] = {
                'time': time_array[idx],
                'temperature': temperature[idx],
                'position': pos_cm
            }
        
        return results
    
    def save_results(self, time_array, temperature, filename='result.csv'):
        """保存结果到CSV文件"""
        df = pd.DataFrame({
            '时间(s)': time_array,
            '温度(摄氏度)': temperature
        })
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"结果已保存到 {filename}")
    
    def plot_temperature_curve(self, time_array, temperature, title="炉温曲线"):
        """绘制炉温曲线"""
        plt.figure(figsize=(12, 8))
        plt.plot(time_array, temperature, 'b-', linewidth=2, label='温度曲线')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('温度 (°C)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加关键温度线
        plt.axhline(y=150, color='g', linestyle='--', alpha=0.7, label='150°C')
        plt.axhline(y=190, color='g', linestyle='--', alpha=0.7, label='190°C')
        plt.axhline(y=217, color='r', linestyle='--', alpha=0.7, label='217°C')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

# 主程序
def main():
    # 创建仿真器
    simulator = ReflowOvenSimulator()
    
    # 尝试加载实验数据
    data_loaded = simulator.load_experimental_data('fujian.xlsx')
    
    if data_loaded:
        # 拟合模型参数
        simulator.fit_parameters()
    else:
        print("使用默认参数进行仿真")
        simulator.heat_transfer_coeff = 0.08  # 默认传热系数
    
    # 问题1的参数设置
    speed = 78  # cm/min
    zone_temps = [173, 198, 230, 257]  # °C
    
    print("\n开始仿真...")
    print(f"传送带速度：{speed} cm/min")
    print(f"温区设定温度：{zone_temps} °C")
    
    # 进行仿真
    time_array, temperature = simulator.simulate_temperature(speed, zone_temps)
    
    if time_array is not None:
        # 获取指定位置的温度
        specific_temps = simulator.get_specific_positions_temp(
            time_array, temperature, speed, zone_temps)
        
        print("\n指定位置的温度：")
        for pos_name, data in specific_temps.items():
            print(f"{pos_name}: {data['temperature']:.2f}°C (时间: {data['time']:.1f}s)")
        
        # 保存结果
        simulator.save_results(time_array, temperature)
        
        # 绘制炉温曲线
        simulator.plot_temperature_curve(time_array, temperature, 
                                        "炉温曲线仿真结果")
        
        print("\n仿真完成！")
    else:
        print("仿真失败！")

if __name__ == "__main__":
    main()