# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: cnn.py

import torch
from torch import nn

if not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock, self).__init__()

        kernel_size = (3,3)
        if isinstance(stride,int):
            stride = (stride,stride)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.rule = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,stride=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rule(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channels,out_channels,stride=1,downsample=None,se=None):
        super(Bottleneck, self).__init__()
        if isinstance(stride,int):
            stride = (stride,stride)

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=(1,1),bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
      

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
  

        self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=(1,1),stride=(1,1),bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
    

        self.rule = nn.ReLU()
       

        # 加入se模块
        self.se = se

        self.downsample = downsample

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
 
        if self.se:
            x = self.se(x)  # downsample不加se

        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.rule(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction):#reduction表示缩减率c/r
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=(1,1), padding=0)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=(1,1), padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class DeepCnn(nn.Module):
    def __init__(self,layers,channels=1,block=None,se_b=4):
        super(DeepCnn, self).__init__()

        if block == 'BasicBlock':
            self.block = BasicBlock
        else:
            self.block = Bottleneck

        # 输入通道
        self.in_channels = 64
        self.out_channels = 64

        self.reduction = 16

        # rpi数据是[-1,1,1,417] 或 [-1,1,22,22]
        self.conv1 = nn.Conv2d(in_channels=channels,out_channels=self.out_channels,kernel_size=7,stride=2,padding=3, bias=False)  # out>[1,64,1,12,12]
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.rule = nn.ReLU(inplace=True)
        # self.rule = nn.PReLU()
        # self.dropout = nn.Dropout(p=0.7)

        self.layer1 = self.make_layer(self.out_channels,layers[0],se_b=se_b)
        self.layer2 = self.make_layer(self.out_channels*2,layers[1],stride=2,se_b=se_b)
        self.layer3 = self.make_layer(self.out_channels*4,layers[2],stride=2,se_b=se_b)
        self.layer4 = self.make_layer(self.out_channels*8,layers[3],stride=2,se_b=se_b)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(self.block.expansion*self.out_channels*8, 2)

    def make_layer(self,out_channel,layer,stride=(1,1),se_b=1):
        # 创建隐藏
        layers = []
        if stride != 1 or self.in_channels != out_channel* self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channel*self.block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*self.block.expansion)
            )

        else:
            downsample = None

        # se的添加位置，【第一，最后，全部】
        se = SEModule(out_channel * self.block.expansion, self.reduction)
        if se_b == 1 or se_b==3:
            layers.append(self.block(self.in_channels, out_channel, stride, downsample=downsample, se=se))
        else:
            layers.append(self.block(self.in_channels, out_channel, stride, downsample=downsample))

        self.in_channels = out_channel * self.block.expansion
        for i in range(1, layer):
            if (i == (layer-1) and se_b==2) or se_b==3:
                layers.extend([self.block(self.in_channels,out_channel,se=se)])
            else:
                layers.extend([self.block(self.in_channels, out_channel)])

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rule(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x,1)
        x = self.fc(x)
        # print(x.size())
        return x


if __name__ == '__main__':
    layer_18 = [2,2,2,2]
    layer_34 = [3,4,6,3]
    layer_50 = [3,4,6,3]
    layer_101 = [3,4,23,3]
    layer_152 = [3,8,36,3]
    block = Bottleneck
    x=torch.Tensor(32, 1536, 13, 107)

    model = DeepCnn(layer_101,channels=x.size(1),block='Bottleneck')
    print(model)
    print(sum([p.nelement() for p in model.parameters()]))
    model(x)

