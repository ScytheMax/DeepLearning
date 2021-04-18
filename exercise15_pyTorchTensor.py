#[1] pages 195f
#[2] folder 07_pytorch

from torch import Tensor

a = Tensor([[1., -3.]])
a.requires_grad = True
print(a.grad)
b = a * a
b.sum().backward()
print(a.grad)