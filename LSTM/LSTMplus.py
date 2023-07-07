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
import tqdm

# 文件读取
def get_Data(path):
    read_data = pd.read_csv(path)
    # 剔除空值
    read_data = read_data[5000:]
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
    label = read_data[['Exits']]  # 取Entries作为标签
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
    #划分数据集
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
def result(x_data, y_data):
    moudle.eval()
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


import  csv
i = 0
with open('stationID.csv','r') as f:
    ids = []
    reader = csv.reader(f)
    for row in reader:
        i+=1
        if(i==1):
            continue
        ids.append(int(row[0]))
# 参数设置
seq_length=8 # 时间步长
input_size=9
num_layers=8
hidden_size=12
batch_size=64
n_iters=10000
lr=0.001
output_size=1
split_ratio=0.8

for id in ids:
    print("这是第{id}站点的训练")
    moudle=Net(input_size,hidden_size,num_layers,output_size,batch_size,seq_length)
    criterion=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(moudle.parameters(),lr=lr)
    # 数据导入
    path = 'Data/station'+str(id)+'.csv'
    # print(path)
    data,label=get_Data(path)
    data,label,mm_y=normalization(data,label)
    x,y=split_windows(data,label,seq_length)
    x_data,y_data,x_train,y_train,x_test,y_test=split_data(x,y,split_ratio)
    train_loader,test_loader,num_epochs=data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size)
    # train
    iter=0
    for epochs in range(num_epochs):
      for i,(batch_x, batch_y) in enumerate (train_loader):
        outputs = moudle(batch_x)
        optimizer.zero_grad()   # 将每次传播时的梯度累积清除
        loss = criterion(outputs,batch_y) # 计算损失
        loss.backward() # 反向传播
        optimizer.step()
        iter+=1
        if iter % 100 == 0:
            print("iter: %d, loss: %1.5f" % (iter, loss.item()))
    torch.save(moudle.state_dict(),"static_dict/zhandianExits"+str(id)+".pth")
    # print("./static_dict/zhandian"+str(id)+".pth")
    moudle.eval()
    result(x_data, y_data)
    result(x_test,y_test)

