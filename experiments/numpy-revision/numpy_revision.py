import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print(f"c shape: {c.shape}")

print(f"a size: {a.size}")
print(f"b size: {b.size}")
print(f"c size: {c.size}")