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

num_epochs = 1
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
        return out, x


def train(train_loader, validate_loader):
    net = Net(input_size, hidden_size, num_classes).to(device)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_step = len(train_loader)
    min_val_loss = float("inf")
    min_loss_epoch = None

    # train
    for epoch in range(num_epochs):
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            net_out, new_features = net(X_train.float())
            loss = criterion(net_out, torch.max(y_train, 1)[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, batch_idx+1, total_step, loss.data/batch_size))

        # validate at each epoch
        with torch.no_grad():
            val_loss = 0
            correct = 0
            for X_val, y_val in validate_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                net_out, new_features = net(X_val.float())
                val_loss += criterion(net_out, torch.max(y_val, 1)[1]).data
                val_pred = net_out.data.max(1)[1]
                correct += val_pred.eq(torch.max(y_val, 1)[1].data).sum()

            val_loss /= len(validate_loader.dataset)
            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                val_loss, correct, len(validate_loader.dataset),
                100. * correct / len(validate_loader.dataset)))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_loss_epoch = epoch
                print('\nSave current best model at epoch {}'.format(epoch))
                torch.save(net.state_dict(),
                           '/home/xuankan/Documents/IR-NN/Checkpoints/model_epoch_{}.ckpt'.format(min_loss_epoch))

    return min_loss_epoch


def test(test_loader, model):
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        test_loss = 0
        correct = 0
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            net_out, new_features = model(X_test.float())
            test_loss += criterion(net_out, torch.max(y_test, 1)[1]).data
            test_pred = net_out.data.max(1)[1]
            correct += test_pred.eq(torch.max(y_test, 1)[1].data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def save_new_feature(train_loader, validate_loader, test_loader, model):
    train_new_features = []
    validate_new_features = []
    test_new_feature = []

    with torch.no_grad():
        for X_train, y_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            train_batch_net_out, train_batch_new_features = model(
                X_train.float())
            train_new_features.append(train_batch_new_features.tolist())

    with torch.no_grad():
        for X_validate, y_validate in validate_loader:
            X_validate = X_validate.to(device)
            y_validate = y_validate.to(device)
            validate_batch_net_out, validate_batch_new_features = model(
                X_validate.float())
            validate_new_features.append(validate_batch_new_features.tolist())

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_batch_net_out, test_batch_new_features = model(X_test.float())
            test_new_features.append(test_batch_new_features.tolist())

    return train_new_features, validate_new_features, test_new_features


if __name__ == "__main__":
    # train dataloader
    train_dataset = Dataset(
        '/home/xuankan/Documents/IR-NN/MQ2007/Fold1/train.txt')
    train_dataset = TensorDataset(train_dataset.X, train_dataset.y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # validate dataloader
    validate_dataset = Dataset(
        '/home/xuankan/Documents/IR-NN/MQ2007/Fold1/vali.txt')
    validate_dataset = TensorDataset(validate_dataset.X, validate_dataset.y)
    validate_loader = DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # test dataloader
    test_dataset = Dataset(
        '/home/xuankan/Documents/IR-NN/MQ2007/Fold1/test.txt')
    test_dataset = TensorDataset(test_dataset.X, test_dataset.y)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    min_loss_epoch = train(train_loader, validate_loader)

    model = Net(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load(
        '/home/xuankan/Documents/IR-NN/Checkpoints/model_epoch_{}.ckpt'.format(min_loss_epoch)))

    test(test_loader, model)

    train_new_features, validate_new_features, test_new_features = save_new_feature(
        train_loader, validate_loader, test_loader, model)
