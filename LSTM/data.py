import csv

import pandas as pd
from tqdm import tqdm
# i=0

# with open('stationID.csv','r') as f:
#     ids = []
#     reader = csv.reader(f)
#     for row in reader:
#         i+=1
#         if(i==1):
#             continue
#         ids.append(int(row[0]))
# print(ids)
#
df_list = [[]]

df = pd.read_csv('./new_traffic_data.csv', index_col='index') # .sort_values(by='Unique ID')
# print(df[df['Unique ID'] == 0])
#
for i in tqdm(range(500)):
    _df = df[df['Unique ID'] == i]
    if len(_df):
        _df.to_csv(f'./station{i}.csv')
# curId = -1
# rows = []
# for row in df.iterrows():
#     if row['Unique ID'] != curId:
#         rows = []
#         if curId !=
#     rows.append(row)
# for row in df.iterrows():
#     df_list
#
# datas = [[]*450]
# i=0
# with open('new_traffic_data.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         i+=1
#         if(i==1):
#             continue
#         datas[int(row[1])].append(row)
# for id in ids:
#     with open("station"+str(id)+".csv","w",newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         #先写入columns_name
#         writer.writerow(['id', 'Unique ID', 'Datetime', 'Structure', 'Neighborhood', 'Entries', 'Exits', 'Year', 'Month',
#                          'Day', 'Hour', 'isHoliday', 'Neighborhood Size'])
#         #写入多行用writerows
#         writer.writerows(datas)