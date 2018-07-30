import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

x_data = np.array([2, 4, 6, 8])
y_data = np.array([4, 8, 12, 16])
x_test = np.array([10, 12, 14, 16])

plt.scatter(x_data, y_data)

w = Variable(torch.Tensor([1.0]), requires_grad = True)
b = Variable(torch.Tensor([1.0]), requires_grad = True)
lr = 0.01

def forward(x):
	return w * x + b

def loss_fn(x, y):
	y_pred = forward(x)
	return (y_pred - y) * (y_pred - y)

def update():
	w.data = w.data - lr * w.grad.data
	b.data = b.data - lr * b.grad.data
	w.grad.data.zero_()
	b.grad.data.zero_()

for epoch in range(100):
	for x_val, y_val in zip(x_data, y_data):
		loss = loss_fn(x_val, y_val)
		loss.backward()
		update()

print w.data[0], b.data[0]

print [(w.data[0] * x + b.data[0]) for x in x_test]

line = [(w.data[0] * x + b.data[0]) for x in x_data]
plt.plot(x_data, line)
plt.show()