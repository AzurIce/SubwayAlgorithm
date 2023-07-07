# -*- coding:UTF-8 -*-
import numpy as np
import torch
from torch import nn
# import matplotlib.pyplot as plt
import torch.utils.data as Data
import csv
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
from torchvision import transforms, datasets
batch_size = 8

def get_Data(seq_length):
    i = 0
    with open('new_traffic_data.csv', 'r', newline='') as csvfile:
        data = []
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            i+=1
            if(i==1):
                continue
            data.append(row)
    data = [[float(row[1]),float(row[3]),[float(row[5])],float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11])] for row in data]
    label = [[float(row[5])] for row in data]  # label就是entries的标签
    #制作时间序列
    x=[]
    y=[]
    for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _x=data[i:i+seq_length]
        _y=label[i+seq_length]
        x.append(_x)
        y.append(_y)
    x,y=np.array(x),np.array(y)
    print('x.shape,y.shape=\n',x.shape,y.shape)
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
    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
    .format(x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape))
    return x_data,y_data,x_train,y_train,x_test,y_test


# 数据装入
def data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size):

    num_epochs=n_iters/(len(x_train)/batch_size) # n_iters代表一次迭代
    num_epochs=int(num_epochs)
    train_dataset=Data.TensorDataset(x_train,y_train)
    test_dataset=Data.TensorDataset(x_test,y_test)
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=True) # 加载数据集,使数据集可迭代
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
    return train_loader,test_loader,num_epochs

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

        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True) # LSTM层
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


seq_length=8 # 时间步长
input_size=9
num_layers=6
hidden_size=12
batch_size=64
n_iters=50
lr=0.001
output_size=1
split_ratio=0.9
moudle=Net(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(moudle.parameters(),lr=lr)
print(moudle)

# 数据导入
x,y=get_Data(seq_length)
x_data,y_data,x_train,y_train,x_test,y_test=split_data(x,y,split_ratio)
train_loader,test_loader,num_epochs=data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size)
# train
# train
iter=0
for epochs in range(num_epochs):
  for i,(batch_x, batch_y) in enumerate (train_loader):
    outputs = moudle(batch_x)
    optimizer.zero_grad()   # 将每次传播时的梯度累积清除
    # print(outputs.shape, batch_y.shape)
    loss = criterion(outputs,batch_y) # 计算损失
    loss.backward() # 反向传播
    optimizer.step()
    iter+=1
    if iter % 100 == 0:
      print("iter: %d, loss: %1.5f" % (iter, loss.item()))