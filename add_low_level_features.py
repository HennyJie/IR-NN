import pandas as pd
import os
from pathlib import Path

index = ["Fold1"]
datasets = ['MQ2007', 'MQ2008']
files = ['test', 'train', 'vali']
files += [f+'_with_newfeatures' for f in files]
files = [f+'.txt' for f in files]
ref = pd.read_csv('low_level_dataset/RefTable.txt', sep="\s+", header=None)
ref.columns = ['id', 'ms_id']

sitemap = pd.read_csv('low_level_dataset/sitemap.txt', sep="\s+", header=None)
sitemap.columns = ['id', 'f1', 'f2', 'f3', 'f4']

for dataset in datasets:
    for fold in index:
        for data_file in files:
            print(Path(dataset)/Path(fold) /
                  Path(data_file))

            data = pd.read_csv(Path(dataset)/Path(fold) /
                               Path(data_file), sep="\s+", header=None)

            m, n = data.shape
            selected_column = list(range(n-9)) + \
                ['f1', 'f2', 'f3', 'f4'] + list(range(n-9, n))
            print(selected_column)

            data = data.merge(ref, left_on=n-7, right_on='ms_id', how='left')
            data = data.merge(sitemap, on='id', how='left')
            data = data.fillna(0)
            for i, col in enumerate(['f1', 'f2', 'f3', 'f4']):

                data[col] = f"{n-10+i}:"+data[col].astype(str)
            data[selected_column].to_csv(Path(dataset)/Path(fold) /
                                         Path('extend_'+data_file), header=None, index=None, sep=' ')
