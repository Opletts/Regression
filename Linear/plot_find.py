import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = np.array([2, 4, 6, 8])
y_data = np.array([4, 8, 12, 16])
x_test = np.array([10, 12, 14, 16])

plt.scatter(x_data, y_data)

mse = []
weight = []
bias = []
for w in np.arange(-10, 10, 0.5):
	for b in np.arange(-10, 10, 0.5):
		loss = 0
		for x_val, y_val in zip(x_data, y_data):
			y_pred = w * x_val + b
			loss += (y_pred - y_val) * (y_pred - y_val)
		mse.append(np.mean(loss))
		weight.append(w)
		bias.append(b)

w = weight[np.argmin(mse)]
b = bias[np.argmin(mse)]

print w, b
line = [(w * x + b) for x in x_data]
print [(w * x + b) for x in x_test]

plt.plot(x_data, line)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter(weight, bias, mse)
ax.set_zlabel('mse')
plt.show()