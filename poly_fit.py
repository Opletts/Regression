import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

x_data = np.array([2, 4, 6, 8])
y_data = np.array([4, 8, 12, 16])
x_test = np.array([10, 12, 14, 16])

plt.scatter(x_data, y_data)

p = Polynomial.fit(x_data, y_data, 1)
pnormal = p.convert(domain=(-1, 1))

print pnormal.coef

plt.plot(*p.linspace())
plt.show()