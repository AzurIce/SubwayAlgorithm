import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.io import Dataset, DataLoader
from paddlenlp.layers.tcn import TemporalBlock
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


# 读取文件，并将文件按时间序列格式化
data = pd.read_csv('new_traffic_data.csv')
# data = data[-10000:]
df_id = data[['Unique ID']]
df_structure = data[['Structure']]
df_nei = data[['Neighborhood Size']]
df_entries = data[['Entries']]
# print(df_entries)
df_exits = data[['Exits']]
df_isHoliday = data[['isHoliday']]

df_entries_np = np.array(df_entries)
df_exits_np = np.array(df_exits)
df_isHoliday_np = np.array(df_isHoliday)
df_id_np = np.array(df_id)
df_structure_np = np.array(df_structure)
df_nei_np = np.array(df_nei)
df = pd.DataFrame(
    df_isHoliday_np.ravel(),
    columns=['isHoliday'],
    index=pd.to_datetime(data['Datetime'])
    #index=pd.date_range(start='2/4/2017 04:00:00', end='8/13/2021 16:00:00', freq='4H')
)


df['Exits'] = df_exits_np.ravel()
df['Unique ID'] = df_id_np.ravel()
df['Structure'] = df_structure_np.ravel()
df['Neighborhood Size'] = df_nei_np.ravel()
df['Entries'] = df_entries_np.ravel()
df[['Exits', 'isHoliday', 'Entries', 'Unique ID', 'Structure', 'Neighborhood Size']]

# 标准化
df[['Exits', 'isHoliday', 'Entries', 'Unique ID', 'Structure', 'Neighborhood Size']] = MinMaxScaler().fit_transform(
    df[['Exits', 'isHoliday', 'Entries', 'Unique ID', 'Structure', 'Neighborhood Size']]
)
std = MinMaxScaler()
df[['Entries']] = std.fit_transform(df[['Entries']])
load_min = std.data_min_
load_max = std.data_max_
load_para = pd.DataFrame({'load_min': load_min, 'load_max': load_max})
# 将处理好的数据存起来
load_para.to_csv('load_para_all.csv', index=False)
df.to_csv('full_data.csv', index=True)
df.loc['2021-5-10 04:00:00':'2021-7-13 00:00:00', :].to_csv('train_data.csv', index=True)
df.loc['2021-7-13 04:00:00':'2021-08-13 16:00:00', :].to_csv('test_data.csv', index=True)

from tqdm import tqdm
class MyDataset(Dataset):
    def __init__(self, mode):
        super(MyDataset, self).__init__()
        if mode == 'train':
            df = pd.read_csv('train_data.csv', parse_dates=[0], index_col=[0])
        elif mode == 'test':
            df = pd.read_csv('test_data.csv', parse_dates=[0], index_col=[0])
        #输入为前8天的数据，这样后面的一天就能获得7天的数据
        time_step_day = 8
        input_time_step = 1 * time_step_day
        output_time_step = 1
        df_len = df.shape[0]
        input_size = df.shape[1]
        data_len = df_len - input_time_step - output_time_step + 1
        self.data = np.zeros((data_len, input_time_step, input_size))
        self.label = np.zeros((data_len, output_time_step))
        for i in tqdm(range(data_len)):
            # print(i)
            self.data[i, :, :] = np.array(df.iloc[i:i + input_time_step,:])
            self.label[i, :] = np.array(df.iloc[i + input_time_step:i + input_time_step + output_time_step, -1])
        print(np.shape(self.data))

    def __getitem__(self, index):
        data = self.data[index, :, :]
        label = self.label[index, :]

        return data, label

    def __len__(self):
        return self.label.shape[0]

# input_channel=5 分别为温度、湿度、降水、风速，流量
# num_channels=[64, 32, 16, 1], kernel_size=3
# 模型用了3个TCN模块
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

# %%
def train(model, datasets, epoch_num):
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    mse_loss = paddle.nn.MSELoss()
    mean_mse_epoch = list()
    for epoch_id in range(epoch_num):
        mse_set = list()
        for batch_id, data in enumerate(datasets()):
            # 准备数据，变得更加简洁
            features, labels = data
            features = paddle.to_tensor(features, dtype='float32')
            labels = paddle.to_tensor(labels, dtype='float32')

            # 前向计算的过程
            predits = model(features)

            # 计算损失，取一个批次样本损失的平均值
            mse = mse_loss(predits, labels)
            mse_set.extend(mse.numpy())

            # 后向传播，更新参数的过程
            mse.backward()
            opt.step()
            opt.clear_grad()
        print("epoch: {}, loss is: {}".format(epoch_id, np.array(mse_set).mean()))
        mean_mse_epoch.append(np.array(mse_set).mean())
    # 保存模型
    paddle.save(model.state_dict(), './load_forcast_station1.pdparams')

    return mean_mse_epoch



# use_gpu = True
# paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

EPOCH_NUM = 10
train_dataset = MyDataset(mode='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
my_model = MyNetwork()
# paddle.summary(my_model)
mean_mse_epoch = train(my_model, train_loader, EPOCH_NUM)
x = [i for i in range(EPOCH_NUM)]
plt.figure()
plt.plot(x, mean_mse_epoch)
plt.show()

DAY = 8800
labList = []
preList = []
# 准备数据
load_para = pd.read_csv('load_para_station1.csv')
load_min = load_para['load_min'].values
load_max = load_para['load_max'].values
print(load_min)
print(load_max)
test_dataset = MyDataset(mode='test')
loss = 0
# 加载模型
my_model = MyNetwork()
my_model.eval()
params_file_path = 'load_forcast_station1.pdparams'
param_dict = paddle.load(params_file_path)
my_model.load_dict(param_dict)
for i in tqdm(range(DAY)):
    features, labels = test_dataset[i] 
    features = paddle.to_tensor(features, dtype='float32')
    features = paddle.unsqueeze(features, axis=0)
    labels = paddle.to_tensor(labels, dtype='float32')
    # 模型预测
    mse_loss = paddle.nn.MSELoss()
    predicts = my_model(features)
    mse = mse_loss(predicts, labels)
    # print('mse:{}'.format(mse.numpy()))
    predicts = predicts.numpy().flatten() * (load_max - load_min) + load_min
    preList.append(predicts)
    labels = labels.numpy() * (load_max - load_min) + load_min
    labList.append(labels)
# 结果可视化
plt.figure()
plt.plot( preList[1:], label='predict')
plt.plot(labList[:-2], label='real')
plt.legend()
plt.show()
for i in range(DAY-1):
    loss+=abs(preList[i+1]-labList[i])
print(loss/DAY)



