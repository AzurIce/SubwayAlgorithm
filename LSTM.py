import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 32
num_epochs = 20
learning_rate = 0.01


class dataset(Data.Dataset):
    def __init__(self, type="train"):
        self.data = data  # data就是n维度向量
        self.label = label  # label就是entries的标签
        self.len = len(data)

    def __getitem__(self, item):
        if type == "train":
            return self.data[item], self.label[item]
        else:
            return (
                self.data[item + int(self.len * 0.8)],
                self.label[item + int(self.len * 0.8)],
            )

    def __len__(self):
        if type == "train":
            return int(self.len * 0.8)
        else:
            return len(self.data) - int(self.len * 0.8)


# 生成训练数据集
train_loader = Data.DataLoader(dataset=dataset(type="train"), batch_size=batch_size)
test_loader = Data.DataLoader(dataset=dataset(type="test"), batch_size=batch_size)


# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(
#     root="../../data/", train=True, transform=transforms.ToTensor(), download=True
# )

# test_dataset = torchvision.datasets.MNIST(
#     root="../../data/", train=False, transform=transforms.ToTensor()
# )

# # Data loader
# train_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset, batch_size=batch_size, shuffle=True
# )

# test_loader = torch.utils.data.DataLoader(
#     dataset=test_dataset, batch_size=batch_size, shuffle=False
# )


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # 此处的-1说明我们只取RNN最后输出的那个hn
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
label = []
output = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        "Test Accuracy of the model on the 10000 test images: {} %".format(
            100 * correct / total
        )
    )

# Save the model checkpoint
torch.save(model.state_dict(), "model.ckpt")
