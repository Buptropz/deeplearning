import torch

#梯度下降法求最优解
x=torch.tensor(10,requires_grad=True,dtype=torch.float32)
y=x**2+20
for i in range(1,1001):
    y=x**2+20
    if x.grad is not None:
        x.grad.zero_()
    y.backward()
    x.data=x.data-0.01*x.grad

print( x)