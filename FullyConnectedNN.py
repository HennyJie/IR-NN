'''
@Author: your name
@Date: 2020-03-21 22:25:32
@LastEditTime: 2020-03-22 13:55:15
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

epochs = 10
batch_size = 200
learning_rate = 0.01
log_interval = 10


class CustomDataset(Dataset):
    def __init__(self, filename):
        with open(filename) as f:
            lines = f.read().split('\n')
        X, y = [], []
        for line in lines:
            X.append(line.split(' ')[2:48])
            y.append(line.split(' ')[0])
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Net(nn.Module):
    def __init__(self, input_dim=46, output_dim=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


def run_nn():
    train_dataset = CustomDataset(
        '/Users/cuihejie/Documents/PhDCourse/IR-NN/MQ2007/Fold1/train.txt')
    test_dataset = CustomDataset(
        '/Users/cuihejie/Documents/PhDCourse/IR-NN/MQ2007/Fold1/test.txt')
    train_loader = DataLoader(train_dataset, batch_size=200, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=200, num_workders=5)

    net = Net()
    print(net)
    if torch.cuda.is_available():
        net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            print(len(data))
            print(target.shape)
            if torch.cuda.is_available():
                data, target = Variable(data.cuda()), Variable(target.cuda())
            else:
                data, target = Variable(data), Variable(target)

            data = data.View(-1, 46)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 46)
        net_out = net(data)
        test_loss += criterion(net_out, target).data[0]
        pred = net_out.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    run_nn()
