'''
@Author: your name
@Date: 2020-03-27 00:49:34
@LastEditTime: 2020-03-28 12:04:02
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /IR-NN/CreateDirectory.py
'''
from pathlib import Path
import os
import sys

output_path = "/Users/cuihejie/Documents/PhDCourse/CS572-InformationRetrieval/output"

datasets = ["MQ2007", "MQ2008"]
folders = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
rankers = ["MART", "LambdaMART"]
metrics = ["NDCG@1", "NDCG@2", "NDCG@3", "NDCG@4", "NDCG@5",
           "NDCG@6", "NDCG@7", "NDCG@8", "NDCG@9", "NDCG@10",
           "P@1", "P@2", "P@3", "P@4", "P@5",
           "P@6", "P@7", "P@8", "P@9", "P@10", "MAP"]

for dataset in datasets:
    for folder in folders:
        for ranker in rankers:
            for metric in metrics:
                new_folder_name = '.'.join(
                    [dataset, folder, ranker, metric, 'output'])
                new_folder_path = os.path.join(output_path, new_folder_name)
                if not os.path.exists(new_folder_path):
                    os.mkdir(new_folder_path)
