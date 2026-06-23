import torch
import numpy as np

#tensor创建
data = torch.tensor([[1,2],[3,4]])

data2 = np.random.randn(2,3)
data2 = torch.tensor(data2)

data3=[[1,2],[3,4]]
data3 = torch.tensor(data3)

#Tensor创建
data4=torch.Tensor(2,3)
data5=torch.Tensor([[1,2],[3,4]])

#基本创建方式(Int Short:int16 Long :int64 Float:float32 Double:float64)
data6=torch.IntTensor(2,3)
data7=torch.IntTensor([2.5,3.3])

#张量类型转换
data8 = torch.full((2,3),5)
data8 = data.type(torch.DoubleTensor)

#张量转换为numpy数组
data_tensor=torch.tensor([2,3,4])
data_numpy=data_tensor.numpy()

#numpy转换为张量
data_numpy=np.array([2,3,4])
data_tensor=torch.from_numpy(data_numpy)

