import math
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_CLASSES = 5
N_SAMPLES = 500
N_FEATURES = 5

# Input Data
X = np.random.randn(N_SAMPLES, N_FEATURES)

# Output Data
y = np.random.randint(0, N_CLASSES, N_SAMPLES)

for i in range(N_CLASSES):
    mask = y == i
    X[mask] += np.random.randn(N_FEATURES) * 2

print(f"input data X shape: {X.shape}")
print(f"Output data y shape: {y.shape}")
print(f"Input data X sample: {X[0]}")
print(f"Output data y sample: {y[:10]}")

class GaussianNaiveBayes:
    def __init__(self):
        self.classes_ = None
        self.means_ = None # shape: (n_classes, n_features)
        self.variances_ = None # shape: (n_classes, n_features)
        self.priors_ = None # shape: (n_classes)

    def fit(self, X, y):
        total_samples = len(X)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.classes_ = np.unique(y)

        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros((n_classes))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]

            self.means_[i] = np.mean(X_cls, axis=0)
            self.variances_[i] = np.var(X_cls, axis=0)
            self.priors_[i] = len(X_cls) / n_samples

        print(f"Means shape: {self.means_.shape}")
        print(f"Variances shape: {self.variances_.shape}")
        print(f"Priors: {self.priors_}")
        print(f"Classes: {self.classes_}")

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        deltas = np.zeros((n_samples, n_classes))

        for idx, class_label in enumerate(self.classes_):
            for j in range(X.shape[1]):
                deltas[:, idx] += (
                    -0.5 * np.log(2*math.pi*self.variances_[idx, j]) -
                    (((X[:, j] - self.means_[idx, j]) ** 2) / (2 * self.variances_[idx, j]))
                )

        return self.classes_[np.argmax(deltas, axis=1)]

GNB = GaussianNaiveBayes()
GNB.fit(X, y)

y_pred = GNB.predict(X)

# Calculate Accuracy
accuracy = np.mean(y_pred == y)
print(f"Accuracy on synthetic data: {accuracy}")