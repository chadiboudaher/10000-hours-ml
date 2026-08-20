import numpy as np

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

N_FEATURES = 7
N_CLASSES = 4
N_SAMPLES = 500

X = np.zeros((N_SAMPLES, N_FEATURES))
# print(X)

y = np.random.randint(0, N_CLASSES, N_SAMPLES)
# print(y)

for i in range(N_SAMPLES):
    if y[i] == 0:
        X[i] = np.random.uniform(0, 1, size=N_FEATURES)
    elif y[i] == 1:
        X[i] = np.random.uniform(1.1, 2.1, size=N_FEATURES)
    elif y[i] == 2:
        X[i] = np.random.uniform(2.2, 3.2, size=N_FEATURES)
    else:
        X[i] = np.random.uniform(3.3, 4.3, size=N_FEATURES)

# print(X[:10])
print(f"First Generated Sample: {X[0]}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {np.unique(y)}")

# Mean
mean = np.zeros((N_SAMPLES))
print(f"Mean vector shape: {mean.shape}")

for i in range(N_SAMPLES):
    feature_mean = 0
    for j in range(N_FEATURES):
        feature_mean += X[i][j]
    feature_mean /= N_FEATURES
    mean[i] = feature_mean

mean_np = np.zeros((N_SAMPLES))
print(f"Mean numpy shape: {mean_np.shape}")

feature_mean_np = 0
for i in range(N_SAMPLES):
    feature_mean_np = 0
    feature_mean_np = np.mean(X[i])
    mean_np[i] = feature_mean_np

print(f"Mean using numpy function: {np.mean(X[1])}")

print(f"Sample mean vector: {mean[:10]}")
print(f"Sample numpy mean vector: {mean_np[:10]}")

class LinearDiscriminantAnalysis:
    def __init__(self):
        ...