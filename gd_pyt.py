import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[2], [4], [6], [8]]))
y_data = Variable(torch.Tensor([[4], [8], [12], [16]]))

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.linear = nn.Linear(1, 1)

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01)
# print net, len(list(net.parameters()))
for epoch in range(100):
	y_pred = net(x_data)
	# print y_pred
	loss = criterion(y_pred, y_data)
	# print loss.data[0]
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

test = Variable(torch.Tensor([[10], [12], [14], [16]]))
# print list(net.parameters())
print net.forward(test).data
