import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.layers.tcn import TemporalBlock
import pymysql
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pymysql
import holidays
import csv
import re
import datetime
from datetime import date
import random

class MyNetwork(nn.Layer):
    def __init__(self, input_channel=6, num_channels=[64,32,8,1], kernel_size=3, dropout=0.2):
        super(MyNetwork, self).__init__()
        layers = nn.LayerList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channel if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout)
                )
            # layers.append(paddle,nn.ReLU())#添加激活函数
        # layers.append(paddle.nn.MultiHeadAttention(64, 1))#多头attention
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        output_time_step = 1
        x_t = x.transpose([0, 2, 1])
        y_t = self.network(x_t)
        output = paddle.squeeze(y_t, axis=1)[:, -output_time_step:] # 只取输出序列后1点
        return output

import csv
def getIDS():
    i = 0
    with open('stationID.csv','r') as f:
        ids = []
        reader = csv.reader(f)
        for row in reader:
            i+=1
            if(i==1):
                continue
            ids.append(int(row[0]))
    return ids

ids = getIDS()
#更新数据库数据
import random
conn = pymysql.connect(
    host='123.60.53.131',
    charset='utf8',
    user='xyh',
    password='2023ShiXun',
    db='NewYork'
)
cursor = conn.cursor()
#删除预测数据库中的数据
cursor.execute('select * from PredictData')
result = cursor.fetchall()
# print(result)
cursor.execute('delete from PredictData')
conn.commit()
sql = 'insert into TrueData values(%s,%s,%s,%s)'
#将预测数据库中的所有数据随机扰动为真实数据
i = 0
for row in result:
    id = row[0]
    dt = row[1]
    Entries = row[2]*(1+random.uniform(-0.5,1))
    Exits = row[3]*(1+random.uniform(-0.5,1))
    i+=1
    cursor.execute(sql,(id,dt,Entries,Exits))
    conn.commit()
print(i)
moudelEntries = MyNetwork()
moudelEntries.eval()
params_file_path = 'load_forcast_entries.pdparams'
param_dict = paddle.load(params_file_path)
moudelEntries.load_dict(param_dict)
moudelExits = MyNetwork()
moudelExits.eval()
params_file_path = 'load_forcast_exits.pdparams'
param_dict = paddle.load(params_file_path)
moudelExits.load_dict(param_dict)
endTime = ''
preTime = ''
for id in ids:
    # 测试所保存的模型，并且获得未来4小时的预测
    cursor = conn.cursor()
    #获得前6天的数据并且预测未来的一天的数据
    sql = "select TEntries,TExits,dateTime from TrueData where ID= %s order by DateTime desc limit 0,6;"
    cursor.execute(sql, id)
    result = cursor.fetchall()
    sql = "select distinct * from stationSize where id = %s "
    cursor.execute(sql,id)
    type = cursor.fetchall()
    struct = type[0][2]
    neighborhoodSize = type[0][1]
    data = []
    dt = ''
    for i in range(6):
        t = result[i][2]
        time_group = re.match(r'(.*)-(.*)-(.*) (.*):(.*):(.*)', t)
        Year = int(time_group.group(1))
        Month = int(time_group.group(2))
        Day = int(time_group.group(3))
        Hour = int(time_group.group(4))
        holiday = date(int(time_group.group(1)), int(time_group.group(2)), int(time_group.group(3))) in holidays.UnitedStates()
        data.append([int(holiday),(result[i][0]-16)/4854,result[i][1]/40697])
        dt = datetime.datetime(Year,Month,Day,Hour)
    endTime = dt
    dt = dt+ datetime.timedelta(hours=4)
    preTime = dt
    data = np.array(data)
    data = paddle.to_tensor(data,dtype='float32')
    data = paddle.unsqueeze(data,axis=0)  
    predictIn = moudelEntries(data)
    miI,maI = 18.0,4335.0
    predictIn = int(predictIn.numpy().flatten()*(maI-miI)+miI+0.5) #数据反归一化
    #用于预测出站的数据
    predictOut = moudelExits(data)
    miI,maI = 0.0,4069.0
    predictOut = int(predictOut.numpy().flatten()*(maI-miI)+miI+0.5) #反归一化
    print("Entries"+str(predictIn))
    print("Exits"+str(predictOut))
    sql = "insert into PredictData values (%s,%s,%s,%s)"
    cursor.execute(sql, (id,str(dt),predictIn,predictOut))
    # 提交事务
sql = "update timeRange set endTime = %s,predictTime =%s"
cursor.execute(sql,(str(endTime),str(preTime)))
cursor.close()
conn.commit()