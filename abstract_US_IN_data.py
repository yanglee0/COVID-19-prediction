import pandas as pd
import numpy as np
import sys
import math
pd.set_option('display.max_columns', None)
# pd.set_option('mode.use_inf_as_na', True)


# 归一化处理
def regularit(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    print('colums', columns)
    for c in columns:
        if (c == 'date' or c == 'location_key'):
            newDataFrame[c] = df[c].tolist()
        else:
            d = df[c]
            MAX = d.max()
            MIN = d.min()
            print(MIN)
            print(MAX)
            newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame


# 流行病 动态
epidemiology_data = pd.read_csv('data_init/epidemiology.csv', low_memory=False)
epidemiology = epidemiology_data[epidemiology_data['location_key'].isin(['US_NY'])]
epidemiology = regularit(epidemiology)
# print(math.isinf(epidemiology))

# 人口统计 静态
demographics_data = pd.read_csv('data_init/demographics.csv', low_memory=False)
demographics = demographics_data[demographics_data['location_key'].isin(['US_NY'])]
# print(len(demographics.columns))
# print(demographics)

# 经济Economy 静态
economy_data = pd.read_csv('data_init/economy.csv', low_memory=False)
economy = economy_data[economy_data['location_key'].isin(['US_NY'])]
# print(len(economy.columns))
# print(economy)

# 健康Health  静态
health_data = pd.read_csv('data_init/health.csv', low_memory=False)
health = health_data[health_data['location_key'].isin(['US_NY'])]
# print(len(health.columns))
# print(health)

# 住院Hospitalizations  动态
hospitalizations_data = pd.read_csv('data_init/hospitalizations.csv', low_memory=False)
hospitalizations = hospitalizations_data[hospitalizations_data['location_key'].isin(['US_NY'])]
hospitalizations = regularit(hospitalizations)
# print(hospitalizations)

# # 流动性Mobility  动态
mobility_data = pd.read_csv('data_init/mobility.csv', low_memory=False)
mobility = mobility_data[mobility_data['location_key'].isin(['US_NY'])]
mobility = regularit(mobility)
# print(mobility)
#
# # 疫苗接种Vaccinations   动态
vaccinations_data = pd.read_csv('data_init/vaccinations.csv', low_memory=False)
vaccinations = vaccinations_data[vaccinations_data['location_key'].isin(['US_NY'])]
vaccinations = regularit(vaccinations)
# print(vaccinations)
#
# # 政府回应Government Response  动态
government_Response_data = pd.read_csv('data_init/oxford-government-response.csv', low_memory=False)
government_Response = government_Response_data[government_Response_data['location_key'].isin(['US_NY'])]
government_Response = regularit(government_Response)
# print(government_Response)
#
# # 天气Weather    动态
weather_data = pd.read_csv('data_init/weather.csv', low_memory=False)
weather = weather_data[weather_data['location_key'].isin(['US_NY'])]
weather = regularit(weather)
# print(weather)

# 地理（备选）Geography
# 年龄（备选）By Age   US没有该部分的数据
# age_data = pd.read_csv('data_init/by-age.csv',low_memory=False)
# age = age_data[age_data['location_key'].isin(['US'])]
# print(age)

# 性别（备选）By Sex  US没有该部分的数据
# sex_data = pd.read_csv('data_init/by-sex.csv',low_memory=False)
# sex = sex_data[sex_data['location_key'].isin(['BR'])]
# print(sex)



# 合并数据
df1 = pd.merge(epidemiology, demographics, on=['location_key'],how ="outer")
# print(df1)
df2 = pd.merge(df1, economy, on=['location_key'],how ="outer")
df3 = pd.merge(df2, health, on=['location_key'],how ="outer")
df4 = pd.merge(df3, hospitalizations, on=['date','location_key'],how ="outer")
df5 = pd.merge(df4, mobility, on=['date','location_key'],how ="outer")
df6 = pd.merge(df5, vaccinations, on=['date','location_key'],how ="outer")
df7 = pd.merge(df6, government_Response, on=['date','location_key'],how ="outer")
df8 = pd.merge(df7, weather, on=['date','location_key'],how ="outer")
# print(df8)

# df.replace([np.inf, -np.inf], np.nan, inplace=True)
df8 = df8.fillna(0.0)
print(df8)
df8.to_csv('US_MD.csv', index=True, header=True)

