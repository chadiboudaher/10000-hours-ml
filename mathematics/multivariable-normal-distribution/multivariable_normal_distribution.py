import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

mu = np.array([0, 0])
sigma = np.array([[1, 0.5], [0.5, 1]])

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

rv = multivariate_normal(mu, sigma)

plt.contourf(X, Y, rv.pdf(pos), levels=20, cmap="viridis")
plt.title("Bivariate Normal Distribution - Contour Plot")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar()
plt.show()