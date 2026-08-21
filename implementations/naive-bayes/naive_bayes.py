import math
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_CLASSES = 5
N_SAMPLES = 500
N_FEATURES = 5

# Input Data
# X = np.random.randn(N_SAMPLES, N_FEATURES)

# Output Data
# y = np.random.randint(0, N_CLASSES, N_SAMPLES)

# for i in range(N_CLASSES):
#     mask = y == i
#     X[mask] += np.random.randn(N_FEATURES) * 2

# print(f"input data X shape: {X.shape}")
# print(f"Output data y shape: {y.shape}")
# print(f"Input data X sample: {X[0]}")
# print(f"Output data y sample: {y[:10]}")

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

        # print(f"Means shape: {self.means_.shape}")
        # print(f"Variances shape: {self.variances_.shape}")
        # print(f"Priors: {self.priors_}")
        # print(f"Classes: {self.classes_}")

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

X, y = make_classification(
    n_samples=1000,
    n_features=7,
    n_classes=4,
    n_clusters_per_class=1,
    n_redundant=0,
    random_state=RANDOM_SEED
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)

GNB_sk = GaussianNB()
GNB_sk.fit(X_train, y_train)

sk_test_pred = GNB_sk.predict(X_test)

sk_test_acc = np.mean(sk_test_pred == y_test)

print(f"sklearn QDA - Test accuracy: {sk_test_acc:.4f}")

GNB = GaussianNaiveBayes()
GNB.fit(X_train, y_train)

train_pred = GNB.predict(X_train)
test_pred = GNB.predict(X_test)

train_acc = np.mean(train_pred == y_train)
test_acc = np.mean(test_pred == y_test)

print(f"Custom QDA - Train accuracy: {train_acc:.4f}")
print(f"Custom QDA - Test accuracy: {test_acc:.4f}")