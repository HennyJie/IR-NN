import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import shutil
import seaborn as sns
import matplotlib.pyplot as plt


folder = 'analysis'


data = defaultdict(list)

file_list = os.listdir(folder)

mdcg = ['NDCG@1', 'NDCG@2', 'NDCG@3', 'NDCG@4', 'NDCG@5',
        'NDCG@6', 'NDCG@7', 'NDCG@8', 'NDCG@9', 'NDCG@10']
p = ['P@1', 'P@2', 'P@3', 'P@4', 'P@5',
     'P@6', 'P@7', 'P@8', 'P@9', 'P@10', 'MAP']
index = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
datasets = ['MQ2007', 'MQ2008']
algorithms = ['Mart', 'LambdaMart']
dfs = {}
for al in algorithms:
    for dataset in datasets:
        dfs[f'{al}_{dataset}'] = pd.DataFrame(columns=p, index=index)
max_int = -1
for file_path in file_list:
    paras = file_path.split('.')
    dataset, fold, algorithm, metric = paras[:4]

    with open(folder/Path(file_path), 'r') as f:
        for line in f.readlines():
            acc = re.findall(r'\((\S+\d+\.\d+)\%\)', line)
            if len(acc) != 0:
                dfs[f'{algorithm}_{dataset}'].loc[fold,
                                                  metric] = float(acc[0])
                max_int = max(max_int, float(acc[0]))

image_path = Path('image')
image_path.mkdir(exist_ok=True, parents=True)


fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
thres = -0.5
fig.set_size_inches(15, 8)
for i, (k, v) in enumerate(dfs.items()):
    v = v.fillna(0)
    # t = v.to_numpy()
    # vmax = max(abs(t.min()), abs(t.max()))
    v[v < thres] = -1
    v[v >= thres] = 1
    print(v)

    cbar = i % 2 != 0
    g = sns.heatmap(v, cmap="Wistia", ax=axs[i//2, i % 2], cbar=cbar)
    g.set_title(k)

plt.tight_layout()
plt.savefig(image_path/Path(f'final.png'))
plt.clf()

plus_count = 0
minus_count = 0
for key, v in data.items():
    data[key] = sum(v)/len(v)
    if sum(v)/len(v) < 0:
        minus_count += 1
    else:
        plus_count += 1
    print(f'{key}\t{data[key]}')
print(minus_count, plus_count)
