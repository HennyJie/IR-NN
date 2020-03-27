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
import pandas as pd
import argparse
from collections import defaultdict
import os
import sys
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 46
num_classes = 3

num_epochs = 100
batch_size = 32
learning_rate = 1e-4
log_interval = 10


class Dataset(Dataset):
    def __init__(self, filename):
        with open(filename) as f:
            lines = f.read().split('\n')[:-1]
        X, y = [], []
        num_in_each_class = defaultdict(int)

        for line in lines:
            X.append([float(item.split(':')[1])
                      for item in line.split(' ')[2:48]])
            for label in line.split(' ')[0]:
                num_in_each_class[label] += 1
                one_hot_label = [0 for _ in range(num_classes)]
                one_hot_label[int(label)] = 1
                y.append(one_hot_label)

        self.X = torch.from_numpy(np.array(X))
        self.y = torch.from_numpy(np.array(y))

        num_most_common = max(num_in_each_class.values())
        num_in_each_class = sorted(
            num_in_each_class.items(), key=lambda d: d[0])

        self.weights = [float(num_most_common/value)
                        for key, value in num_in_each_class]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Net(nn.Module):
    def __init__(self, input_size=46, num_classes=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out, x


def train(train_dataset, validate_dataset):
    train_tensor_dataset = TensorDataset(train_dataset.X, train_dataset.y)
    train_loader = DataLoader(
        train_tensor_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    validate_tensor_dataset = TensorDataset(
        validate_dataset.X, validate_dataset.y)
    validate_loader = DataLoader(
        validate_tensor_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    net = Net(input_size, num_classes).to(device)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_class_weights = torch.FloatTensor(train_dataset.weights).cuda()
    train_criterion = nn.CrossEntropyLoss(weight=train_class_weights)

    val_class_weights = torch.FloatTensor(validate_dataset.weights).cuda()
    val_criterion = nn.CrossEntropyLoss(weight=val_class_weights)

    total_step = len(train_loader)
    min_val_loss = float("inf")
    min_loss_epoch = None

    # train
    for epoch in range(num_epochs):
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            net_out, new_features = net(X_train.float())
            loss = train_criterion(net_out, torch.max(y_train, 1)[
                1])

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
                val_loss += val_criterion(net_out, torch.max(y_val, 1)
                                          [1]).data
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
                model_path = os.path.join(
                    sys.path[0], 'Checkpoints', f'model_epoch_{min_loss_epoch}.ckpt')
                torch.save(net.state_dict(), model_path)

    return min_loss_epoch


def test(test_dataset, model):
    test_tensor_dataset = TensorDataset(test_dataset.X, test_dataset.y)
    test_loader = DataLoader(
        test_tensor_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    class_weights = torch.FloatTensor(test_dataset.weights).cuda()
    test_criterion = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        test_loss = 0
        correct = 0
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            net_out, new_features = model(X_test.float())
            test_loss += test_criterion(net_out, torch.max(y_test, 1)
                                        [1]).data
            test_pred = net_out.data.max(1)[1]
            print("test_pred: ", test_pred)
            correct += test_pred.eq(torch.max(y_test, 1)[1].data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def save_new_feature(train_dataset, validate_dataset, test_dataset, model):
    train_new_features = []
    validate_new_features = []
    test_new_features = []

    train_dataset = TensorDataset(train_dataset.X, train_dataset.y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    validate_dataset = TensorDataset(validate_dataset.X, validate_dataset.y)
    validate_loader = DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = TensorDataset(test_dataset.X, test_dataset.y)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    with torch.no_grad():
        for X_train, y_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            train_batch_net_out, train_batch_new_features = model(
                X_train.float())
            for new_feature in train_batch_new_features.tolist():
                train_new_features.append(new_feature)

    with torch.no_grad():
        for X_validate, y_validate in validate_loader:
            X_validate = X_validate.to(device)
            y_validate = y_validate.to(device)
            validate_batch_net_out, validate_batch_new_features = model(
                X_validate.float())
            for new_feature in validate_batch_new_features.tolist():
                validate_new_features.append(new_feature)

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_batch_net_out, test_batch_new_features = model(X_test.float())
            for new_feature in test_batch_new_features.tolist():
                test_new_features.append(new_feature)

    train_new_features = np.array(train_new_features)
    validate_new_features = np.array(validate_new_features)
    test_new_features = np.array(test_new_features)

    train_new_features = (
        train_new_features - train_new_features.min(axis=0)) / train_new_features.max(axis=0)
    validate_new_features = (
        validate_new_features - validate_new_features.min(axis=0)) / validate_new_features.max(axis=0)
    test_new_features = (
        test_new_features - test_new_features.min(axis=0)) / test_new_features.max(axis=0)

    return train_new_features, validate_new_features, test_new_features


def write_new_features_to_file(train_new_features, train_data_path, validate_new_features, validate_data_path,
                               test_data_path, test_new_features):
    train_add_newfeatures_path = os.path.split(
        train_data_path)[0] + '/train_with_newfeatures.txt'
    validate_add_newfeatures_path = os.path.split(
        validate_data_path)[0] + '/vali_with_newfeatures.txt'
    test_add_newfeatures_path = os.path.split(
        test_data_path)[0] + '/test_with_newfeatures.txt'

    data = pd.read_csv(train_data_path, sep="\s+", header=None)
    for i in range(8):
        tmp = list(map(lambda x: str(i+47)+':'+f"{x:.6f}",
                       train_new_features[:, i]))
        data.insert(i+48, str(i), tmp)
    data.to_csv(train_add_newfeatures_path, sep=' ', header=False, index=False)

    data = pd.read_csv(validate_data_path, sep="\s+", header=None)
    for i in range(8):
        tmp = list(map(lambda x: str(i+47)+':'+f"{x:.6f}",
                       validate_new_features[:, i]))
        data.insert(i+48, str(i), tmp)
    data.to_csv(validate_add_newfeatures_path,
                sep=' ', header=False, index=False)

    data = pd.read_csv(test_data_path, sep="\s+", header=None)
    for i in range(8):
        tmp = list(map(lambda x: str(i+47)+':'+f"{x:.6f}",
                       test_new_features[:, i]))
        data.insert(i+48, str(i), tmp)
    data.to_csv(test_add_newfeatures_path, sep=' ', header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train NN to get new features.')
    parser.add_argument('--datasets', metavar='datasets', type=str, nargs='+',
                        help='Datasets to be processed')
    parser.add_argument('--folders', metavar='folders', type=str, nargs='+',
                        help='Folds to be processed')
    args = parser.parse_args()
    datasets = args.datasets
    folders = args.folders
    print(datasets)
    print(folders)

    for dataset in datasets:
        for folder in folders:
            print("dataset: ", dataset)
            print("folder: ", folder)
            train_data_path = os.path.join(
                sys.path[0], dataset, folder, 'train.txt')
            validate_data_path = os.path.join(
                sys.path[0], dataset, folder, 'vali.txt')
            test_data_path = os.path.join(
                sys.path[0], dataset, folder, 'test.txt')

            # train dataloader
            train_dataset = Dataset(train_data_path)

            # validate dataloader
            validate_dataset = Dataset(validate_data_path)

            # test dataloader
            test_dataset = Dataset(test_data_path)

            min_loss_epoch = train(train_dataset, validate_dataset)
            model = Net(input_size, num_classes).to(device)
            model_path = os.path.join(
                sys.path[0], 'Checkpoints', f'model_epoch_{min_loss_epoch}.ckpt')
            model.load_state_dict(torch.load(model_path))

            test(test_dataset, model)

            train_new_features, validate_new_features, test_new_features = save_new_feature(
                train_dataset, validate_dataset, test_dataset, model)

            write_new_features_to_file(
                train_new_features, train_data_path, validate_new_features, validate_data_path,
                test_data_path, test_new_features)
