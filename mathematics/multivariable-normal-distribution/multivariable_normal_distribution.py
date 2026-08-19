import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# mu = np.array([0, 0])
# sigma = np.array([[1, 0.5], [0.5, 1]])

# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x, y)
# pos = np.dstack((X, Y))

# rv = multivariate_normal(mu, sigma)

# plt.contourf(X, Y, rv.pdf(pos), levels=20, cmap="viridis")
# plt.title("Bivariate Normal Distribution - Contour Plot")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.colorbar()
# plt.show()

def plot_3d_surface(rv, ax):
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = rv.pdf(pos)
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

def plot_contour(rv, ax):
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = rv.pdf(pos)
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(contour, ax=ax)

mu = [0, 0]

cov_spherical = [[1, 0], [0, 1]]
rv_spherical = multivariate_normal(mu, cov_spherical)

cov_diagonal = [[1, 0], [0, 2]]
rv_diagonal = multivariate_normal(mu, cov_diagonal)

cov_full = [[1, 0.8], [0.8, 1]]
rv_full = multivariate_normal(mu, cov_full)

fig = plt.figure(figsize=(18, 12))

ax1 = fig.add_subplot(231, projection='3d')
plot_3d_surface(rv_spherical, ax1)
ax1.set_title("Spherical Covariance (3D)")

ax2 = fig.add_subplot(234)
plot_contour(rv_spherical, ax2)
ax2.set_title("Spherical Covariance (Contour)")

ax3 = fig.add_subplot(232, projection='3d')
plot_3d_surface(rv_diagonal, ax3)
ax3.set_title("Diagonal Covariance (3D)")

ax4 = fig.add_subplot(235)
plot_contour(rv_diagonal, ax4)
ax4.set_title("Diagonal Covariance (Contour)")

ax5 = fig.add_subplot(233, projection='3d')
plot_3d_surface(rv_full, ax5)
ax5.set_title("Full Covariance (3D)")

ax6 = fig.add_subplot(236)
plot_contour(rv_full, ax6)
ax6.set_title("Full Covariance (Contour)")

plt.tight_layout()
plt.show()