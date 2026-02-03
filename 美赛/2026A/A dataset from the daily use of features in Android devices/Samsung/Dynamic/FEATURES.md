基本信息
device_id：设备唯一标识符。
timestamp：采样时间戳，格式一般为 YYYY-MM-DD HH:MM:SS（本地时间）。
collected：采集标记（常见为 1=成功/有效，0=未采到/无效）。
foreground_app：前台应用标识符（正整数类型）。

屏幕与亮度
screen_status：屏幕状态编码（0 = 熄屏，1 = 亮屏）。
bright_level：屏幕亮度等级（0–255，数值越大越亮）。

连接与硬件开关
bluetooth：蓝牙状态编码（-1 = 不可用，0 = 断开连接，1 = 正在连接，2 = 已连接，3 = 正在断开，10 = 关闭，11 = 正在开启，12 = 开启，13 = 正在关闭）。
gps_status：GPS 开关状态编码（0 = 关闭，1 = 开启）。
gps_activity：GPS 使用状态（0 = 未使用，3 = 正在使用）。
saving_mode：省电模式开关（0 = 开启，1 = 关闭）。
flashlight：手电筒状态（0 = 关闭，1 = 开启）。
airplane_mode：飞行模式开关（0 = 关闭，1 = 开启）。
fingerprint：指纹功能可用状态（0 = 可用，1 = 不可用）。

电池
battery_level：电量百分比（0–100）。
battery_health：电池健康状态编码（-1 = 未知，0 = 过冷，1 = 放电完毕，2 = 良好，3 = 过热，4 = 过压）。
battery_charging_status：充电状态编码（1 = 未知，2 = 充电中，3 = 放电中，4 = 已充满）。
battery_connection_status：充电连接方式（0 = 电池供电，1 = 交流充电，2=USB 充电）。
battery_temperature：电池温度（摄氏度，浮点型）。
battery_current：电池电流（mA）。
battery_voltage：电池电压（V）。
battery_power：电池瞬时功率（瓦特 / 秒）。

网络与通信
network_mode：网络连接状态编码（0 = 无连接，1 = 有连接）。
mobile_mode：移动网络制式编码（"0"= 移动网络关闭，"2G"/"3G"/"4G"/"5G"= 对应网络制式）。
mobile_status：移动网络连接状态编码（0 = 未知，1 = 已连接）。
mobile_roaming：是否漫游（0 = 未漫游，1 = 已漫游）。
mobile_rx：移动网络当前接收速率（Mbps）。
mobile_tx：移动网络当前发送速率（Mbps）。
wifi_status：Wi‑Fi 连接状态编码（0 = 断开连接，1 = 已连接）。
wifi_intensity：Wi‑Fi 信号强度（dB，分贝）。
wifi_speed：Wi‑Fi 链路速率（Mbps）。
wifi_rx：Wi‑Fi 当前接收速率（Mbps）。
wifi_tx：Wi‑Fi 当前发送速率（Mbps）。

声音
ring_mode：铃声模式编码（0 = 静音，1 = 静音 + 振动，2 = 响铃 + 振动）。
sound_level：系统音量等级（0–15，数值越大音量越大）。
playback_status：音频播放状态（0 = 无媒体播放，1 = 媒体正在播放）。
内存与存储
ram_usage：RAM 已使用量（字节）。
ram_free：RAM 可用量（字节）。
rom_usage：存储（ROM）已使用量（字节）。
rom_free：存储（ROM）可用量（字节）。
CPU 与温度
cpu_usage：CPU 使用率（百分比，浮点型）。
cpu_temperature：CPU 温度（摄氏度，浮点型）。
frequency_core0：CPU 第 0-5 核当前使用频率（浮点型）。
frequency_core6：CPU 第 6-7 核当前使用频率（浮点型）。













