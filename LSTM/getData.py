from datetime import date

import torch
import torch.nn as nn
import re

import seaborn as sns
import numpy as np
import holidays
import pandas as pd
import matplotlib.pyplot as plt

traffic_path = '../data/NYC_subway_traffic_2017-2021.csv'
neighborhood_path = '../data/bigSmall.csv'

# 获取数据
traffic_data = pd.read_csv(traffic_path)
neighborhood_data = pd.read_csv(neighborhood_path)

# 获取美国的节假日
us_holidays = holidays.UnitedStates()

# 数据清洗
traffic_data = traffic_data[['Unique ID', 'Datetime', 'Structure', 'Neighborhood', 'Entries', 'Exits']]
# traffic_data =traffic_data[0: 500]
year_list = []
month_list = []
day_list = []
hour_list = []
isHoliday_list = []
struct_list = []
size_list = []
for row in traffic_data.itertuples():
    # 通过getattr(row, ‘name')获取元素
    time_group = re.match(r'(.*)-(.*)-(.*) (.*):(.*):(.*)', getattr(row, 'Datetime'))
    # 处理时间
    year_list.append(time_group.group(1))
    month_list.append(time_group.group(2))
    day_list.append(time_group.group(3))
    hour_list.append(time_group.group(4))
    '''traffic_data['Year'] = time_group.group(1)
    traffic_data['Month'] = time_group.group(2)
    traffic_data['Day'] = time_group.group(3)
    traffic_data['Hour'] = time_group.group(4)'''

    # 检查某一天是否是节假日
    is_holiday = date(int(time_group.group(1)), int(time_group.group(2)), int(time_group.group(3))) in us_holidays
    if is_holiday:
        isHoliday_list.append(1)
    else:
        isHoliday_list.append(0)

    # 记录社区大小
    data = neighborhood_data[neighborhood_data['place'] == getattr(row, 'Neighborhood')]
    if data.empty:
        size_list.append(None)
    else:
        size_list.append(data['type'].iloc[0])
    # print(traffic_data['Neighborhood Size'])

    # 获取structure的结构信息
    stru = getattr(row,'Structure')
    if(stru=='Subway'):
        struct_list.append(1)
    elif(stru=='Elevated'):
        struct_list.append(2)
    elif(stru=='Open Cut'):
        struct_list.append(3)
    elif (stru == 'Viaduct'):
        struct_list.append(4)
    elif (stru == 'At Grade'):
        struct_list.append(5)

traffic_data['Year'] = year_list
traffic_data['Month'] = month_list
traffic_data['Day'] = day_list
traffic_data['Hour'] = hour_list
traffic_data['isHoliday'] = isHoliday_list
traffic_data['Neighborhood Size'] = size_list
traffic_data['Structure'] = struct_list

traffic_data.to_csv(r'../data/new_traffic_data.csv')



# 数据分析
# data.info()
# data.head()
# multivariate_analysis(num=500)
# print(data.corr()['实时客流人数'].abs().sort_values(ascending=False))
'''
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size'''
'''
data = data[data['Unique ID'] == 1]
data = data[0:24]
print(data)'''


'''
plt.title('Passenger')
plt.ylabel('EntriesPassengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(data['Datetime'], data['Entries'])
plt.show()'''