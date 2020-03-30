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

num_epochs = 10
batch_size = 32
learning_rate = 1e-4
log_interval = 10
num_new_features = 8


def square(filename):
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

    data = np.power(np.array(X), 2)
    return data


def write_new_features_to_file(train_new_features, train_data_path, train_type):
    train_add_newfeatures_path = os.path.split(
        train_data_path)[0] + f'/square_{train_type}_with_newfeatures.txt'
    data = pd.read_csv(train_data_path, sep="\s+", header=None)
    m, n = data.shape
    _, k = train_new_features.shape
    for i in range(k):
        tmp = list(map(lambda x: str(n-10+i)+':'+f"{x:.6f}",
                       train_new_features[:, i]))
        data.insert(n-9+i, str(n-9+i+100), tmp)
    data.to_csv(train_add_newfeatures_path, sep=' ', header=False, index=False)


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

            train_new_features = square(train_data_path)

            validate_new_features = square(validate_data_path)

            test_new_features = square(test_data_path)

            write_new_features_to_file(
                train_new_features, train_data_path, 'train')
            write_new_features_to_file(
                validate_new_features, validate_data_path, 'vali')
            write_new_features_to_file(
                test_new_features, test_data_path, 'test')
