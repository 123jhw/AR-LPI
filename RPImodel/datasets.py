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
import pandas as pd
import torch

from copy import deepcopy
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.combine import SMOTETomek
from torch.utils.data import Dataset, TensorDataset


class RPIseqDataset(Dataset):
    def __init__(self,file_path,conv,model_type=False):
        if not os.path.exists(file_path):
            raise RuntimeError('Dataset file_path not found.:{}'.format(file_path))
        self.data = pd.read_csv(file_path)
        self.columns = self.data.columns.tolist()

        if 'Y' in self.columns:
            x = self.data.shape[-1] - 1
            self.target = self.data['Y']
            self.target_transform = torch.tensor(self.target.values, dtype=torch.long)
            del self.data['Y']
            self.is_label = True
        else:
            self.is_label = False
            x = self.data.shape[-1]

        if conv:
            t = ['t_{}'.format(i) for i in range(57)]
            self.data[t] = 0.0
            size = (-1,1,22,22)
        else:
            size = (-1,1,1,x)

        self.features = self.data
        print(self.features.shape)

        self.transform = torch.tensor(self.features.values,dtype=torch.float32)
        self.transform = self.StandardScaler()
        self.transform = torch.nn.functional.normalize(self.transform, p=2, dim=1, eps=1e-12, out=None)

        if model_type == "cnn":
            self.transform = self.transform.view(size)

        del self.data

    def __getitem__(self, idx):
        # 返回数据集deepcopy
        data = self.transform[idx]
        if self.is_label:
            label = self.target_transform[idx]
            return deepcopy(data), deepcopy(label)
        else:
            return deepcopy(data)

    def __len__(self):
        # 返回数据集大小
        size = self.target_transform.size()[0]
        return size

    def normalization(self,x):
        # 归一化 0-1
        amin, amax = x.min(), x.max()  # 求最大最小值
        inputs = (x - amin) / (amax - amin)  # (矩阵元素-最小值)/(最大值-最小值)
        return inputs

    def StandardScaler(self):
        x = self.transform
        mean, std = x.mean(), x.std()
        inputs = (x - mean) / std
        return inputs

    def MinMaxScaler(self,x):
        amin, amax = x.min(), x.max()  # 求最大最小值
        X_std = (x - amin) / (amax - amin)
        X_scaled = X_std * (amax - amin) + amin
        return X_scaled


class SMOTEData:
    def __init__(self,path):
        self.data = pd.read_csv(path)
        self.features = self.data.iloc[:,1:]
        self.target = self.data.iloc[:,0]
        self.random_state = 33

    def smote(self,X,y):
        # '过采样'
        sos = SMOTE(random_state=self.random_state)
        X,y = sos.fit_resample(X,y)
        return X,y

    def random(self,X, y):
        # 随机采样
        ros = RandomOverSampler(random_state=self.random_state,sampling_strategy='auto')
        X, y = ros.fit_resample(X, y)
        return X, y

    def combine(self,X,y):
        # 综合采样
        cos = SMOTETomek(random_state=self.random_state)
        X, y = cos.fit_resample(X, y)
        return X, y


