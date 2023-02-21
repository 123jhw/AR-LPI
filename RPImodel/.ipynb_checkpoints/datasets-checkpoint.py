# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: datasets.py
# @Author: derek
# @Institution: --- BeiJing, ---, China
# @E-mail: derek_yyl@outlook.com
# @Site: 自定义数据集
# @Time: 11/8/22 10:29 上午
# ---
import os
import random
from copy import deepcopy
from typing import Optional

import pandas as pd
import numpy as np
import torch
from collections import Counter

from torch._utils import _accumulate
from torch.utils.data import Dataset, DataLoader, Subset


class RPIseqDataset(Dataset):
    def __init__(self,file_path,conv,model_type=False):
        if not os.path.exists(file_path):
            raise RuntimeError('Dataset file_path not found.:{}'.format(file_path))
        self.data = pd.read_csv(file_path)
        x = self.data.shape[-1]
        if conv:
            t = ['t_{}'.format(i) for i in range(57)]
            self.data[t] = 0.0
            size = (-1,1,22,22)
        else:
            size = (-1,1,1,x-1)

        self.features = self.data.iloc[:,1:]
        print(self.features.shape)
        self.target = self.data.iloc[:,0]

        self.transform = torch.tensor(self.features.values,dtype=torch.float32)
        self.target_transform = torch.tensor(self.target.values,dtype=torch.long)

        self.transform = self.StandardScaler(self.transform)
        self.transform = torch.nn.functional.normalize(self.transform, p=2, dim=1, eps=1e-12, out=None)

        if model_type=="cnn":
            self.transform = self.transform.view(size)

        del self.data

    def __getitem__(self, idx):
        # 返回数据集deepcopy
        data = self.transform[idx]
        label = self.target_transform[idx]
        return deepcopy(data), deepcopy(label)

    def __len__(self):
        # 返回数据集大小
        size = self.target_transform.size()[0]
        return size

    def normalization(self,x):
        # 归一化 0-1
        amin, amax = x.min(), x.max()  # 求最大最小值
        inputs = (x - amin) / (amax - amin)  # (矩阵元素-最小值)/(最大值-最小值)
        return inputs

    def StandardScaler(self,x):
        mean, std = x.mean(), x.std()
        inputs = (x - mean) / std
        return inputs

    def MinMaxScaler(self,x):
        amin, amax = x.min(), x.max()  # 求最大最小值
        X_std = (x - amin) / (amax - amin)
        X_scaled = X_std * (amax - amin) + amin
        return X_scaled

def _random_split(dataset,lengths):
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    # 单独处理一下类别
    label_1 = torch.where(dataset.target_transform == 0)[0]
    label_2 = torch.where(dataset.target_transform == 1)[0]

    indices = torch.randperm(sum(lengths), generator=None).tolist()
    print(len(indices))
    i,j = 0,0
    total_data = []
    for offset, length in zip(_accumulate(lengths), lengths):
        t1 = int(length // 2)
        t2 = length - t1
        a = label_1[i:t1+i].tolist()
        b = label_2[j:t2+j].tolist()
        print(len(a),len(b))
        i += t1
        j += t2
        # print(indices[offset - length: offset])
        total_data.append(torch.utils.data.Subset(dataset,a+b))
    return total_data
    # return [torch.utils.data.Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def random_split(dataset, lengths, generator: Optional[torch.Generator]= torch.default_generator):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    # indices = torch.randperm(sum(lengths), generator=generator).tolist()
    indices = dataset.indices
    random.shuffle(indices)
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


if __name__ == '__main__':
    path = "/Users/derek/Downloads/RPIseq_new_12388.csv"
    load_data = RPIseqDataset(path,True)
    # data,label = next(iter(load_data))
    sample = load_data.target_transform.size(0)
    train_size = int(sample * 0.8)
    test_size = sample - train_size
    train_dataset, test_dateset = torch.utils.data.random_split(load_data, [train_size, test_size])

    list1 = test_dateset.indices
    list2 = train_dataset.indices
    print(len(list1),len(list2))
    _y = []
    train_data = DataLoader(train_dataset, batch_size=128, shuffle=True)
    for i in range(6):
        total = 0
        a = []

        print(set(list1) & set(list2))
        # train_dataset, test_dateset = random_split(train_dataset, [train_size, 0])



        for x,y in train_data:
            c = set(y) & set(_y)


            _y = y
            break
        # print(Counter(np.array(a)))




