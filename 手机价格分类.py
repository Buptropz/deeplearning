import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from torchsummary import summary


def create_data():
    data=pd.read_csv('data/手机价格预测.csv')
    print(data.shape)

    #获取x特征列 和 y标签列
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]

    #特征列转为浮点型
    x=x.astype(np.float32)

    #切分训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)

    #转为张量数据集
    train_data=TensorDataset(torch.tensor(x_train.values),torch.tensor(y_train.values))
    test_data=TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))

    #返回结果
    return train_data,test_data,x_train.shape[1],len(np.unique(y))

class phoneprice(nn.Module):
    def __init__(self,input_dim,output_dim):
        #1.1 继承父类
        super().__init__()
        #1.2 定义网络结构
        self.hidden1=nn.Linear(input_dim,128)
        self.hidden2=nn.Linear(128,256)
        self.output=nn.Linear(256,output_dim)

    #1.3 定义前向传播
    def forward(self,x):
        x=self.hidden1(x)
        x=torch.relu(x)

        x=self.hidden2(x)
        x=torch.relu(x)

        x=self.output(x)
        return x



if __name__ == '__main__':
    #1.创建数据集
    train_dateset,test_dataset,input_dim,output_dim=create_data()
    #2.创建数据加载器
    model = phoneprice(input_dim, output_dim)
    #3.计算模型参数
    summary(model,input_size=(16,input_dim))