'''
@Author: your name
@Date: 2020-03-21 22:25:32
@LastEditTime: 2020-03-22 15:54:28
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /IR-NN/FullyConnectedNN.py
'''
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 46
hidden_size = 128
num_classes = 3

num_epochs = 10
batch_size = 32
learning_rate = 1e-4
log_interval = 10


class Dataset(Dataset):
    def __init__(self, filename):
        with open(filename) as f:
            lines = f.read().split('\n')[:-1]
        X, y = [], []
        for line in lines:
            X.append([float(item.split(':')[1])
                      for item in line.split(' ')[2:48]])
            for label in line.split(' ')[0]:
                one_hot_label = [0 for _ in range(num_classes)]
                one_hot_label[int(label)] = 1
                y.append(one_hot_label)

        self.X = torch.from_numpy(np.array(X))
        self.y = torch.from_numpy(np.array(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Net(nn.Module):
    def __init__(self, input_size=46, hidden_size=128, num_classes=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


def run_nn():
    train_dataset = Dataset(
        '/home/xuankan/Documents/IR-NN/MQ2007/Fold1/train.txt')
    train_dataset = TensorDataset(train_dataset.X, train_dataset.y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = Dataset(
        '/home/xuankan/Documents/IR-NN/MQ2007/Fold1/test.txt')
    test_dataset = TensorDataset(test_dataset.X, test_dataset.y)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    net = Net(input_size, hidden_size, num_classes).to(device)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            net_out = net(data.float())
            loss = criterion(net_out, torch.max(target, 1)[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, batch_idx+1, total_step, loss.item()))

    with torch.no_grad():
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            net_out = net(data.float())
            test_loss += criterion(net_out, torch.max(target, 1)[1]).data
            pred = net_out.data.max(1)[1]
            correct += pred.eq(torch.max(target, 1)[1].data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    torch.save(net.state_dict(), 'model.ckpt')


if __name__ == "__main__":
    run_nn()
