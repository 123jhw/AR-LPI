# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: autocode.py
import torch
from torch import nn

if not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'


class AutoEncodeLinear(torch.nn.Module):
    def __init__(self,in_features, out_features=512,layer=3, in_actName='Tanh', out_actName='Tanh'):
        super(AutoEncodeLinear, self).__init__()

        if in_actName == "Tanh":
            act_fn = nn.Tanh()
        elif in_actName == "ReLU":
            act_fn = nn.ReLU()
        else:
            act_fn = nn.LeakyReLU()

        if out_actName == "Tanh":
            out_fn = nn.Tanh()
        elif in_actName == "ReLU":
            out_fn = nn.ReLU()
        else:
            out_fn = nn.Sigmoid()

        self.num = in_features
        self.out = out_features
        # encode = nn.Sequential()
        # decode = nn.Sequential()
        encode = []
        decode = []
        for i in range(layer):
            encode.append(nn.Linear(self.num,self.out))
            encode.append(act_fn)
            decode.append(act_fn)
            decode.append(nn.Linear(self.out,self.num))
            self.num,self.out = self.out,self.out // 2

        decode[0] = out_fn
        self.encode = nn.Sequential(*encode[:-1]) #封装模块
        self.decode = nn.Sequential(*decode[::-1])

    def forward(self, x):
        # print(x.is_cuda)
        encode = self.encode(x)
        decode = self.decode(encode)
        return encode,decode


class AutoEncodeCnn(torch.nn.Module):
    def __init__(self, n_features,dim,device,ndf=128,layer=5,out_actName='Tanh',in_actName='ReLU',max_pool=True):
        super(AutoEncodeCnn, self).__init__()

        self.device = device
        self.ndf = ndf
        self.layer = layer
        self.dim = dim
        self.n_features = n_features

        if in_actName == "Tanh":
            act_fn = nn.Tanh()
        elif in_actName == "ReLU":
            act_fn = nn.ReLU()
        else:
            act_fn = nn.LeakyReLU()

        if out_actName == "Tanh":
            out_fn = nn.Tanh()
        elif in_actName == "ReLU":
            out_fn = nn.ReLU()
        else:
            out_fn = nn.Sigmoid()

        self.encode, self.decode = self.create(act_fn, out_fn, max_pool,n_features)

    def create(self,act_fn,out_fn,max_pool,n_features):
        encode = []
        decode = []

        in_num = 1

        if n_features == 427:
            kernel_size = (1,3)
            max_pool_size = (1,2)
        else:
            kernel_size = 3
            max_pool_size = 2

        for i in range(1,self.layer+1):
            # 编码
            encode.append(nn.Conv2d(in_channels=in_num, out_channels=i*self.ndf, kernel_size=kernel_size, stride=1, padding=1))
            encode.append(nn.BatchNorm2d(i*self.ndf))
            encode.append(act_fn)
            # 解码
            decode.append(act_fn)
            decode.append(nn.BatchNorm2d(in_num))
            decode.append(nn.ConvTranspose2d(in_channels=i*self.ndf,out_channels=in_num,kernel_size=kernel_size,stride=1,padding=1))
            in_num = i*self.ndf
        # 池化
        if max_pool:
            # 第一次特征降维
            encode.append(nn.MaxPool2d(kernel_size=max_pool_size, stride=max_pool_size))
            decode.append(nn.Upsample(scale_factor=max_pool_size, mode='nearest'))

        if self.dim == 2 and self.n_features == 427:
            # 这个是针对输入: input=[none,1,1,427],第二次特征降维
            encode.append(
                nn.Conv2d(in_channels=in_num, out_channels=(self.layer + 1) * self.ndf, kernel_size=kernel_size, stride=1,
                          padding=1))
            encode.append(nn.BatchNorm2d((self.layer + 1) * self.ndf))
            encode.append(act_fn)
            decode.append(act_fn)
            decode.append(nn.BatchNorm2d(in_num))
            decode.append(nn.ConvTranspose2d(in_channels=(self.layer + 1) * self.ndf, out_channels=in_num, kernel_size=kernel_size,stride=1, padding=1))

            encode.append(nn.MaxPool2d(kernel_size=max_pool_size, stride=max_pool_size))
            decode.append(nn.Upsample(scale_factor=max_pool_size, mode='nearest'))

        decode[0] = out_fn
        decode = decode[::-1] #倒序排列
        return nn.Sequential(*encode),nn.Sequential(*decode)

    def forward(self, x):
        batch,features =x.size()[0],x.size()[-1]
        if self.n_features == 427:
            b = torch.zeros(batch,1,1,1,device=self.device)
            x = torch.cat([x,b],dim=3)
        encode = self.encode(x)
        decode = self.decode(encode)
        if self.n_features == 427:
            decode, _ = torch.split(decode,[features,1],dim=3)
        return encode,decode
