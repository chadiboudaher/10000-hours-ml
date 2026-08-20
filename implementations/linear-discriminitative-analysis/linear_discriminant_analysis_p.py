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

class0_num = class1_num = class2_num = class3_num = 0
for i in range(N_SAMPLES):
    if y[i] == 0:
        X[i] = np.random.uniform(0, 1, size=N_FEATURES)
        class0_num +=1
    elif y[i] == 1:
        X[i] = np.random.uniform(1.1, 2.1, size=N_FEATURES)
        class1_num +=1
    elif y[i] == 2:
        X[i] = np.random.uniform(2.2, 3.2, size=N_FEATURES)
        class2_num +=1
    else:
        X[i] = np.random.uniform(3.3, 4.3, size=N_FEATURES)
        class3_num +=1

# Number of samples per class
print(f"Number samples for class 0: {class0_num}")
print(f"Number samples for class 1: {class1_num}")
print(f"Number samples for class 2: {class2_num}")
print(f"Number samples for class 3: {class3_num}")

# Divide Sample Data per class
X_0 = np.zeros((class0_num, N_FEATURES))
X_1 = np.zeros((class1_num, N_FEATURES))
X_2 = np.zeros((class2_num, N_FEATURES))
X_3 = np.zeros((class3_num, N_FEATURES))

idx_0 = idx_1 = idx_2 = idx_3 = 0

for i in range(N_SAMPLES):
    if y[i] == 0:
        X_0[idx_0] = X[i]
        idx_0 += 1
    elif y[i] == 1:
        X_1[idx_1] = X[i]
        idx_1 += 1
    elif y[i] == 2:
        X_2[idx_2] = X[i]
        idx_2 += 1
    else:
        X_3[idx_3] = X[i]
        idx_3 += 1

# Print sample data from each class
print(f"Sample data class 1: {X_0[0]}")


# print(X[:10])
# print(f"First Generated Sample: {X[0]}")
# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}")
# print(f"Classes: {np.unique(y)}")

# Mean
mean = np.zeros((N_SAMPLES))
# print(f"Mean vector shape: {mean.shape}")

for i in range(N_SAMPLES):
    feature_mean = 0
    for j in range(N_FEATURES):
        feature_mean += X[i][j]
    feature_mean /= N_FEATURES
    mean[i] = feature_mean

mean_np = np.zeros((N_SAMPLES))
# print(f"Mean numpy shape: {mean_np.shape}")

feature_mean_np = 0
for i in range(N_SAMPLES):
    feature_mean_np = 0
    feature_mean_np = np.mean(X[i])
    mean_np[i] = feature_mean_np

# print(f"Mean using numpy function: {np.mean(X[1])}")

# print(f"Sample mean vector: {mean[:10]}")
# print(f"Sample numpy mean vector: {mean_np[:10]}")

# Mean per class
# class_mean = np.

class LinearDiscriminantAnalysis:
    def __init__(self):
        ...