import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
import pymysql
import holidays
import  csv
import re
import  time
import datetime
from datetime import date
import random

def get_Data(path):
    read_data = pd.read_csv(path)
    # 剔除空值
    # read_data = read_data[:1000]
    read_data = read_data.dropna()
    read_data = read_data[[
        'Structure',
        'Entries',
        'Exits',
        'Year',
        'Month',
        'Day',
        'Hour',
        'isHoliday',
        'Neighborhood Size']]  # 以十个特征作为数据
    label = read_data[['Entries']]  # 取Entries作为标签
    print(label.__len__)
    return read_data, label


# 数据预处理
def normalization(data,label):
    mm_x=MinMaxScaler() # 导入sklearn的预处理容器
    mm_y=MinMaxScaler()
    data=data.values    # 将pd的系列格式转换为np的数组格式
    label=label.values
    data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    label=mm_y.fit_transform(label)
    return data,label,mm_y


# 时间向量转换
def split_windows(data,label,seq_length):
    x=[]
    y=[]
    for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _x=data[i:(i+seq_length),:]
        _y=label[i+seq_length]
        x.append(_x)
        y.append(_y)
    x,y=np.array(x),np.array(y)
    y.reshape(-1,1)
    return x,y


# 数据分离
def split_data(x,y,split_ratio):
    train_size=int(len(y)*split_ratio)
    test_size=len(y)-train_size
    x_data=Variable(torch.Tensor(np.array(x)))
    y_data=Variable(torch.Tensor(np.array(y)))
    x_train=Variable(torch.Tensor(np.array(x[0:train_size])))
    y_train=Variable(torch.Tensor(np.array(y[0:train_size])))
    y_test=Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    x_test=Variable(torch.Tensor(np.array(x[train_size:len(x)])))

    # print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
    # .format(x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape))

    return x_data,y_data,x_train,y_train,x_test,y_test


# 数据装入
def data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size):
    num_epochs=n_iters/(len(x_train)/batch_size) # n_iters代表一次迭代
    print(num_epochs)
    num_epochs=int(num_epochs)
    train_dataset=Data.TensorDataset(x_train,y_train)
    test_dataset=Data.TensorDataset(x_train,y_train)
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=True) # 加载数据集,使数据集可迭代
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
    return train_loader,test_loader,num_epochs

def result(module,x_data, y_data):
    moudle.eval()
    print(x_data.shape)
    train_predict = moudle(x_data)
    data_predict = train_predict.data.numpy()
    y_data_plot = y_data.data.numpy()
    y_data_plot = np.reshape(y_data_plot, (-1,1))
    data_predict = mm_y.inverse_transform(data_predict)
    y_data_plot = mm_y.inverse_transform(y_data_plot)

    plt.plot(y_data_plot)
    plt.plot(data_predict)
    plt.legend(('real', 'predict'),fontsize='15')
    plt.show()
    print('MAE/RMSE')
    print(mean_absolute_error(y_data_plot, data_predict))
    print(np.sqrt(mean_squared_error(y_data_plot, data_predict)))
# 定义一个类
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,seq_length) -> None:
        super(Net,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.num_directions=1 # 单向LSTM

        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=0.2) # LSTM层
        # self.drop = Dropout(0.2)
        self.fc=nn.Linear(hidden_size,output_size) # 全连接层

    def forward(self,x):
        # e.g.  x(10,3,100) 三个句子，十个单词，一百维的向量,nn.LSTM(input_size=100,hidden_size=20,num_layers=4)
        # out.shape=(10,3,20) h/c.shape=(4,b,20)
        batch_size, seq_len = x.size()[0], x.size()[1]    # x.shape=(604,3,3)
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0)) # output(5, 30, 64)
        pred = self.fc(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred

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
seq_length=6 # 时间步长
input_size=9
num_layers=8
hidden_size=12
output_size=1
batch_size=64

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

for id in ids:
    # 测试所保存的模型，并且获得未来4小时的预测
    cursor = conn.cursor()
    path = "./static_dict/zhandian1.pth"
    m_state_dict = torch.load(path)
    moudle=Net(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    moudle.load_state_dict(m_state_dict)
    #获得前6天的数据并且预测未来的一天的数据
    sql = "select TEntries,TExits,dateTime from TrueData where ID= %s order by DateTime desc limit 0,6;"
    cursor.execute(sql, id)
    result = cursor.fetchall()
    sql = "select distinct * from stationSize where id = %s "
    cursor.execute(sql,id)
    type = cursor.fetchall()
    # print(id)
    struct = type[0][2]
    neighborhoodSize = type[0][1]
    data = []
    maI = result[0][0]
    miI = result[0][0]
    miO = result[0][1]
    maO = result[0][1]
    dt = ''
    for i in range(seq_length):
        maI = max(maI,result[i][0])
        miI = min(miI,result[i][0])
        maO = max(maO,result[i][1])
        miO = min(miO,result[i][1])
        t = result[i][2]
        time_group = re.match(r'(.*)-(.*)-(.*) (.*):(.*):(.*)', t)
        Year = int(time_group.group(1))
        Month = int(time_group.group(2))
        Day = int(time_group.group(3))
        Hour = int(time_group.group(4))
        holiday = date(int(time_group.group(1)), int(time_group.group(2)), int(time_group.group(3))) in holidays.UnitedStates()
        data.append([struct,result[i][0],result[i][1],Year,Month,Day,Hour,int(holiday),neighborhoodSize])
        dt = datetime.datetime(Year,Month,Day,Hour)
    dt = dt+ datetime.timedelta(hours=4)
    data = [data]
    data = np.array(data)
    x_data=Variable(torch.Tensor(np.array(data)))
    predict = moudle(x_data)
    predictIn = float(predict.data.numpy()[0][0])
    predictIn = int(predictIn*(maI-miI)/2+miI+0.5)
    # #用于预测出站的数据
    outPath = "./static_dict/zhandianExits1.pth"
    m_state_dict = torch.load(outPath)
    moudle.load_state_dict(m_state_dict)
    predict = moudle(x_data)
    predictOut = float(predict.data.numpy()[0][0])
    predictOut = int(predictOut*(maO-maO)/2+maO+0.5)
    # print("Entries"+str(predictIn))
    # print("Exits"+str(predictOut))
    sql = "insert into PredictData values (%s,%s,%s,%s)"
    cursor.execute(sql, (id,str(dt),predictIn,predictOut))
    # 提交事务
    cursor.close()
    conn.commit()
