import numpy as np

# Basic shape operations

a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print(f"c shape: {c.shape}")

print(f"a size: {a.size}")
print(f"b size: {b.size}")
print(f"c size: {c.size}")

# Reshaping Arrays

X = np.random.randn(100, 64)
print(f"Original: {X.shape}")

X_batch = X.reshape(10, 10, 64)
print(f"Batch reshape: {X_batch.shape}")

X_flat = X.reshape(100, -1)
print(f"Flattened: {X_flat.shape}")

X_cnn = X.reshape(100, 8, 8, 1)
print(f"CNN format: {X_cnn.shape}")

# Broadcasting Rules Practice

X = np.random.randn(10, 5)
w = np.random.randn(5, 1)

y = X @ w
print(f"Matrix multiplication: {y.shape}")

bias = np.array([0.5])
y_with_bias = y + bias
print(f"With bias: {y_with_bias.shape}")

A = np.random.randn(10, 5)
B = np.random.randn(5,)
result = A + B
print(f"Broadcasted: {result.shape}")  # (10, 5)

try:
    C = np.random.randn(10, 5)
    D = np.random.randn(3,)
    result = C + D
except ValueError as e:
    print(f"Error: {e}")

X_train = np.random.randn(800, 20)
X_test = np.random.randn(200, 20)

X_train_batch = np.expand_dims(X_train, axis=0)
X_test_batch = np.expand_dims(X_test, axis=0)

print(f"Original X_train shape: {X_train.shape}")
print(f"batch X_train shape: {X_train_batch.shape}")

print(f"Original X_test shape: {X_test.shape}")
print(f"batch X_test shape: {X_test_batch.shape}")