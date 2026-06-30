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

def train(train_dataset,input_dim,output_dim):
    #创建数据加载器
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    #创建模型对象
    model1=phoneprice(input_dim,output_dim)
    #定义损失函数和优化器
    criterion=nn.CrossEntropyLoss ()
    optimizer=optim.SGD(model1.parameters(),lr=0.01)
    #开始训练
    for epoch in range(50):
        total_loss=0.0
        batch_num = 0
        start=time.time()
        for x,y in train_loader:
            model1.train()
            y_pred=model1(x)
            loss=criterion(y_pred,y)
            #优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #累计损失
            total_loss+=loss.item()
            batch_num+=1

        print(f'epoch:{epoch + 1},loss:{total_loss / batch_num:.4f},time:{time.time() - start:.2f}s')
    #保存模型(保存模型参数)
    #参数: 模型参数,保存路径
    print(f'\n\n模型参数学习：{model1.state_dict()}\n\n')
    torch.save(model1.state_dict(),'./model/手机价格分类.pth')

def evalaute(test_dataset,input_dim,output_dim):\
    #创建神经网络
    model1=phoneprice(input_dim,output_dim)
    #加载模型参数
    model1.load_state_dict(torch.load('./model/手机价格分类.pth'))
    #创建数据加载器
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False)
    #定义变量
    correct=0
    #测试
    for x,y in test_loader:
        #模型测试
        model1.eval()
        #模型预测
        y_pred=model1(x)
        print(y_pred)
        #根据加权求和，获取预测结果
        y_pred=torch.argmax(y_pred,dim=1)
        print(f'y_pred:{y_pred}')


if __name__ == '__main__':
    #1.创建数据集
    train_dateset,test_dataset,input_dim,output_dim=create_data()
    #2.创建数据加载器
    model = phoneprice(input_dim, output_dim)
    #3.计算模型参数
    #summary(model,input_size=(16,input_dim))
    #4.训练模型
    train(train_dateset,input_dim,output_dim)
    #模型的测试
    evalaute(test_dataset,input_dim,output_dim)