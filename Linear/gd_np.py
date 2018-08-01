import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

x_data = np.array([2, 4, 6, 8])
y_data = np.array([4, 8, 12, 16])
x_test = np.array([10, 12, 14, 16])

plt.scatter(x_data, y_data)

w = 1
b = 1
lr = 0.01

def forward(x):
	return w * x + b

def loss_fn(y_pred, y):
	return np.power(y_pred - y, 2)

def update(x, y, y_pred):
	global w, b
	w = w - lr * 2 * x * (y_pred - y)
	b = b - lr * 2 * (y_pred - y)

for epoch in range(100):
	for x_val, y_val in zip(x_data, y_data):
		y_pred = forward(x_val)
		loss = loss_fn(y_pred, y_val)
		update(x_val, y_val, y_pred)
		# print loss

print w, b
print [(w * x + b) for x in x_test]

line = [(w * x + b) for x in x_data]
plt.plot(x_data, line)
plt.show()