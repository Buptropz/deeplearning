import torch
import torch.nn as nn
#模型可视化
from torchsummary import summary

class Model(nn.Module):
    #1.初始化父类成员 及 搭建神经网络
    def __init__(self):
        #1.1 初始化父类成员
        super().__init__()
        #1.2 搭建隐藏层+输出层
        #隐藏层1：输入特征为3，输出特征为3
        self.hidden1=nn.Linear(3,3)
        #隐藏层2：输入特征为3，输出特征为2
        self.hidden2=nn.Linear(3,2)
        #输出层：输入特征为2，输出特征为2
        self.output=nn.Linear(2,2)

        #1.3 隐藏层参数初始化
        nn.init.xavier_normal(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)

        nn.init.kaiming_normal(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)

    #2.前向传播
    def forward(self,x):
        #分解版
        x=self.hidden1(x)
        x=torch.sigmoid(x)
        #合并版
        x=torch.sigmoid(self.hidden1(x))
        x=torch.relu(self.hidden2(x))
        x=torch.softmax(self.output(x),dim=-1)

        return x

#模型训练
def train():
    #创建模型对象
    my_model=Model()
    #创建数据样本
    data=torch.randn(size=(5,3))
    #调用神经网络，进行模型训练
    output=my_model(data)
    print(output)
    print(output.shape)
    print(output.requires_grad)

    #查看模型参数
    summary(my_model,(5,3))
    print(my_model.parameters())
    for name,param in my_model.named_parameters():
        print(f'name:{name}')
        print(f'param:{param}')


if __name__=='__main__':
    train()