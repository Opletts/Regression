import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([2, 4, 6, 8])
y_data = np.array([4, 8, 12, 16])
x_test = np.array([10, 12, 14, 16])

plt.scatter(x_data, y_data)

w = (np.mean(x_data) * np.mean(y_data) - np.mean(x_data * y_data)) / (np.mean(x_data) * np.mean(x_data) - np.mean(x_data * x_data))
b = np.mean(y_data) - w * np.mean(x_data)

print w, b
print [(w * x + b) for x in x_test]

line = [(w * x + b) for x in x_data]

plt.plot(x_data, line)
plt.show()