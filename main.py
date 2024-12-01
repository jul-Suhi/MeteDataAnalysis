import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 数据读取
data = pd.read_csv('weather_data.csv')

# 查看数据基本信息
print("数据前几行展示：")
print(data.head())
print("数据基本信息：")
print(data.info())
print("数据数值型列统计信息：")
print(data.describe())


# 处理缺失值
data = data.dropna()

# 确保日期列的数据类型为日期时间类型
data['date'] = pd.to_datetime(data['date'])

# 处理异常值
lower_bound = data['temperature'].quantile(0.05)
upper_bound = data['temperature'].quantile(0.95)
data.loc[(data['temperature'] < lower_bound) | (data['temperature'] > upper_bound), 'temperature'] = np.nan
data['temperature'] = data['temperature'].fillna(data['temperature'].mean())

# 数据分析与可视化

# 1. 气温随时间变化趋势分析及可视化
daily_mean_temperature = data.groupby('date')['temperature'].mean()

# 使用matplotlib绘制折线图展示气温变化趋势
plt.figure(figsize=(10, 6))
plt.plot(daily_mean_temperature.index, daily_mean_temperature.values)
plt.xlabel('Date')
plt.ylabel('Average Temperature (°C)')
plt.title('Average Temperature Trend over Time')
plt.xticks(rotation=45)
plt.show()

# 使用seaborn绘制，更加美观简洁
plt.figure(figsize=(10, 6))
sns.lineplot(x=daily_mean_temperature.index, y=daily_mean_temperature.values)
plt.xlabel('Date')
plt.ylabel('Average Temperature (°C)')
plt.title('Average Temperature Trend over Time')
plt.xticks(rotation=45)
plt.show()

# 2. 降水量与气温的相关性分析及可视化
# 计算降水量和气温的相关系数
correlation = data['precipitation'].corr(data['temperature'])
print(f"Correlation between precipitation and temperature: {correlation}")

# 使用散点图展示两者关系
plt.figure(figsize=(8, 6))
sns.scatterplot(x='temperature', y='precipitation', data=data)
plt.xlabel('Temperature (°C)')
plt.ylabel('Precipitation (mm)')
plt.title('Relationship between Precipitation and Temperature')
plt.show()

# 3. 不同季节的气象特征分析
def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

data['season'] = data['date'].apply(get_season)

# 分析不同季节的平均气温
seasonal_mean_temperature = data.groupby('season')['temperature'].mean()
print(seasonal_mean_temperature)

# 可视化不同季节的平均气温对比（使用柱状图）
plt.figure(figsize=(8, 6))
sns.barplot(x=seasonal_mean_temperature.index, y=seasonal_mean_temperature.values)
plt.xlabel('Season')
plt.ylabel('Average Temperature (°C)')
plt.title('Average Temperature by Season')
plt.show()

# 4. 风速分布分析及可视化
plt.figure(figsize=(8, 6))
sns.histplot(data['wind_speed'], bins=10)
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Frequency')
plt.title('Distribution of Wind Speed')
plt.show()