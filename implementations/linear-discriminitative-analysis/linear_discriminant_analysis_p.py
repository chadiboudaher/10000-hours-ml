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
print(f"Sample data class 0: {X_0[0]}")
print(f"Sample data class 1: {X_0[1]}")
print(f"Sample data class 2: {X_0[2]}")
print(f"Sample data class 3: {X_0[3]}")

print("=================================================")

print(f"Shape of X_0: {X_0.shape}")
print(f"Shape of X_0.T: {X_0.T.shape}")
print(f"Mean of X_0.T: {np.mean(X_0.T)}")
print(f"Mean of X_0 with axis=0: {np.mean(X_0, axis=0)}")

print("=================================================")

# Mean per class
class_mean = np.zeros((N_CLASSES, N_FEATURES))
# for i in range(N_CLASSES):
class_mean[0] = np.mean(X_0, axis=0)
class_mean[1] = np.mean(X_1, axis=0)
class_mean[2] = np.mean(X_2, axis=0)
class_mean[3] = np.mean(X_3, axis=0)

print(f"Class mean shape: {class_mean.shape}")
print(class_mean)


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

for i in range(N_SAMPLES):
    feature_mean_np = 0
    feature_mean_np = np.mean(X[i])
    mean_np[i] = feature_mean_np

# print(f"Mean using numpy function: {np.mean(X[1])}")

# print(f"Sample mean vector: {mean[:10]}")
# print(f"Sample numpy mean vector: {mean_np[:10]}")

# Prior
prior = np.zeros((N_CLASSES))
classes_num = [class0_num, class1_num, class2_num, class3_num]

for i in range(N_CLASSES):
    prior[i] = classes_num[i] / N_SAMPLES

# print(f"The prior value for class 0: {prior[0]}")
# print(f"The prior value for class 1: {prior[1]}")
# print(f"The prior value for class 2: {prior[2]}")
# print(f"The prior value for class 3: {prior[3]}")
# print(f"Sum of priors: {np.sum(prior)}")

# Covariance
cov = np.zeros((N_CLASSES, N_FEATURES, N_FEATURES))

cov[0] = (X_0 - class_mean[0]).T @ (X_0 - class_mean[0]) / class0_num
cov[1] = (X_1 - class_mean[1]).T @ (X_1 - class_mean[1]) / class1_num
cov[2] = (X_2 - class_mean[2]).T @ (X_2 - class_mean[2]) / class2_num
cov[3] = (X_3 - class_mean[3]).T @ (X_3 - class_mean[3]) / class3_num

print(f"Cov[0] shape: {cov[0].shape}")
print(f"Cov[1] shape: {cov[1].shape}")
print(f"Cov[2] shape: {cov[2].shape}")
print(f"Cov[3] shape: {cov[3].shape}")
# print(f"Covariance Matrix: {cov}")

pooled_cov = (
    (class0_num/N_SAMPLES) * cov[0] + 
    (class1_num/N_SAMPLES) * cov[1] + 
    (class2_num/N_SAMPLES) * cov[2] + 
    (class3_num/N_SAMPLES) * cov[3]
)

print(f"pooled cov shape: {pooled_cov.shape}")
print(f"pooled covariance: {pooled_cov}")

# Inverse Covariance
inv_pooled_cov = np.linalg.inv(pooled_cov)

print(f"Inverse pooled covariance shape: {inv_pooled_cov.shape}")
print(f"Inverse pooled covariance: {inv_pooled_cov}")

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.means_ = None
        self.prior_ = None
        self.pooled_covariance_ = None
        self.classes_ = None
        self.inv_pooled_cov_ = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.classes_ = np.unique(y)
        classes = np.unique(y)
        counts = np.zeros((n_classes))

        for i in range(n_samples):
            cls_index = np.where(classes == y[i])[0][0]
            counts[cls_index] += 1

        max_samples = np.max(counts)
        X_by_class = np.zeros((n_classes, max_samples, n_features))

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        row_trackers = np.zeros(n_classes)

        for i in range(n_samples):
            cls = y[i]
            cls_idx = class_to_idx[cls]

            X_by_class[cls_idx][row_trackers[cls_idx]] = X[i]
            row_trackers[cls_idx] += 1

        self.means_ = np.zeros((n_classes, n_features))

        for i in range(n_classes):
            self.means_[i] = np.mean(X_by_class[i][:counts[i]], axis=0)

        self.prior_ = np.zeros((n_classes))

        for i in range(n_classes):
            self.prior_[i] = counts[i] / n_samples

        cov = np.zeros((n_classes, n_features, n_features))

        for i in range(n_classes):
            cov[i] = (
                (X_by_class[i][:counts[i]] - self.means_[i]).T @ 
                (X_by_class[i][:counts[i]] - self.means_[i]) / counts[i]
            )

        reg = 1e-6
        self.pooled_covariance_ = np.zeros((n_features, n_features))
        self.pooled_covariance_ += reg * np.eye(n_features)

        for i in range(n_classes):
            self.pooled_covariance_ += (counts[i] / n_samples) * cov[i]

        self.inv_pooled_cov_ = np.linalg.inv(self.pooled_covariance_)

        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)

        for i in range(n_classes):
            self.coef_[i] = self.inv_pooled_cov_ @ self.means_[i]
            self.intercept_[i] = (
                (-1 / 2) * self.means_[i].T @ 
                self.inv_pooled_cov_ @ 
                self.means_[i] + np.log(self.prior_[i])
            )