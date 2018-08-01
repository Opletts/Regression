import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[2], [4], [6], [8]]))
y_data = Variable(torch.Tensor([[0], [0], [1], [1]]))

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.linear = nn.Linear(1, 1)

	def forward(self, x):
		lnr = self.linear(x)
		y_pred = F.sigmoid(lnr)

		return y_pred

net = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01)

for epoch in range(10000):
	y_pred = net(x_data)
	loss = criterion(y_pred, y_data)
	print loss.data[0]
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

test = Variable(torch.Tensor([[4]]))
print net(test).data[0][0] > 0.5