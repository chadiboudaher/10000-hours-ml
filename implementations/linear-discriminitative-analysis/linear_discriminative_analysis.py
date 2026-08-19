import numpy as np

RANDOM_SEED = 42

N_FEATURES = 4
N_SAMPLES = 100
N_CLASSES = 2
NOISE = 0.1

np.random.seed(RANDOM_SEED)

# Simulate data for LDA

X = np.random.rand(N_SAMPLES, N_FEATURES)
y = np.random.randint(0, N_CLASSES, N_SAMPLES)

for i in range(N_SAMPLES):
    if y[i] == 0:
        X[i] = np.random.uniform(0, 1) + NOISE
    else:
        X[i] = np.random.uniform(2, 3) + NOISE
print(X)
print(y)

class LinearDiscriminantAnalysis:
    def __init__(self):
        ...