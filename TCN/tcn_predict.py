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
    def __init__(self, input_channel=3, num_channels=[64, 32, 8, 1], kernel_size=3, dropout=0.2):
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
        output = paddle.squeeze(y_t, axis=1)[:, -output_time_step:]  # 只取输出序列后1点
        return output


import csv


def getIDS():
    i = 0
    with open('stationID.csv', 'r') as f:
        ids = []
        reader = csv.reader(f)
        for row in reader:
            i += 1
            if (i == 1):
                continue
            ids.append(int(row[0]))
    return ids


ids = getIDS()
# 更新数据库数据
import random

conn = pymysql.connect(
    host='123.60.53.131',
    charset='utf8',
    user='xyh',
    password='2023ShiXun',
    db='NewYork'
)
cursor = conn.cursor()
'''
# 删除预测数据库中的数据
cursor.execute('select * from PredictData')
result = cursor.fetchall()
# print(result)
cursor.execute('delete from PredictData')
# conn.commit()
sql = 'insert into TrueData values(%s,%s,%s,%s)'
# 将预测数据库中的所有数据随机扰动为真实数据
i = 0
for row in result:
    id = row[0]
    dt = row[1]
    Entries = row[2] * (1 + random.uniform(-0.2, 0.2))
    Exits = row[3] * (1 + random.uniform(-0.2, 0.2))
    i += 1
    cursor.execute(sql, (id, dt, Entries, Exits))
    # conn.commit()
'''

sql = "UPDATE TrueData SET isPredict = 0 WHERE STR_TO_DATE(dateTime, '%Y-%m-%d %H:%i:%s') <= NOW()"
cursor.execute(sql)
conn.commit()

entries_pre_path = '../model/entries/model_station'
exits_pre_path = '../model/exits/model_station'
for id in ids:
    '''
    if id == 11:
        break'''
    # 加载进站模型
    modelEntries = MyNetwork()
    modelEntries.eval()
    params_entries_path = entries_pre_path + str(id) + '.pdparams'
    param_dict = paddle.load(params_entries_path)
    modelEntries.load_dict(param_dict)
    # 加载出站模型
    modelExits = MyNetwork()
    modelExits.eval()
    params_exits_path = exits_pre_path + str(id) + '.pdparams'
    param_dict = paddle.load(params_exits_path)
    modelExits.load_dict(param_dict)

    endTime = ''
    preTime = ''

    # 测试所保存的模型，并且获得未来4小时的预测
    cursor = conn.cursor()
    # 获得前8天的数据并且预测未来的一天的数据
    sql = "select TEntries,TExits,dateTime from TrueData where ID= %s order by DateTime desc limit 0,8;"
    cursor.execute(sql, id)
    result = cursor.fetchall()
    # 逆序
    result = tuple(reversed(result))
    # print(type(result))
    # print(result)
    '''
    sql = "select distinct * from stationSize where id = %s "
    cursor.execute(sql, id)
    type = cursor.fetchall()
    struct = type[0][2]
    neighborhoodSize = type[0][1]'''
    dataIn = []
    dataOut = []
    dt = ''

    for i in range(8):
        t = result[i][2]
        time_group = re.match(r'(.*)-(.*)-(.*) (.*):(.*):(.*)', t)
        Year = int(time_group.group(1))
        Month = int(time_group.group(2))
        Day = int(time_group.group(3))
        Hour = int(time_group.group(4))
        holiday = date(int(time_group.group(1)), int(time_group.group(2)),
                       int(time_group.group(3))) in holidays.UnitedStates()
        dataIn.append([int(holiday), result[i][1], result[i][0]])
        dataOut.append([int(holiday), result[i][0], result[i][1]])
        dt = datetime.datetime(Year, Month, Day, Hour)
    endTime = dt
    dt = dt + datetime.timedelta(hours=4)
    preTime = dt
    print('id = ', id)
    print('preTime = ', preTime)
    # 用于预测进站的数据
    dataIn = np.array(dataIn)
    dataIn = paddle.to_tensor(dataIn, dtype='float32')
    dataIn = paddle.unsqueeze(dataIn, axis=0)
    predictIn = modelEntries(dataIn)
    predictIn = int(predictIn.numpy().flatten())
    # 用于预测出站的数据
    dataOut = np.array(dataOut)
    dataOut = paddle.to_tensor(dataOut, dtype='float32')
    dataOut = paddle.unsqueeze(dataOut, axis=0)
    predictOut = modelExits(dataOut)
    predictOut = int(predictOut.numpy().flatten())

    print("Entries = " + str(predictIn))
    print("Exits = " + str(predictOut))
    sql = "insert into TrueData values (%s,%s,%s,%s, 1)"
    cursor.execute(sql, (id, str(dt), predictIn, predictOut))
# 提交事务
# sql = "update timeRange set endTime = %s,predictTime =%s"
# cursor.execute(sql, (str(endTime), str(preTime)))
cursor.close()
conn.commit()
