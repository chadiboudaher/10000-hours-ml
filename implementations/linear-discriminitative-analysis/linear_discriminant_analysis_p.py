import numpy as np

RANDOM_SEED = 42

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
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {np.unique(y)}")

class LinearDiscriminantAnalysis:
    def __init__(self):
        ...