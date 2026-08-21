import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SKQDA
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

N_FEATURES = 7
N_CLASSES = 4
N_SAMPLES = 500

# X = np.random.randn(N_SAMPLES, N_FEATURES)
# y = np.random.randint(0, N_CLASSES, N_SAMPLES)

# for i in range(N_CLASSES):
#     mask = y == i
#     X[mask] += np.random.randn(N_FEATURES) * 2

class QuadraticDiscriminantAnalysis:
    """
    This algorithm is similar to LDA with a little adjustments.
    """
    def __init__(self):
        self.means_ = None
        self.prior_ = None
        self.covariance_ = None
        self.classes_ = None
        self.inv_cov_ = None
        self.log_det_ = None
        # self.intercept_ = None
        # self.coef_ = None
        self.score_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.classes_ = np.unique(y)
        classes = np.unique(y)
        counts = np.zeros((n_classes), dtype=int)

        for i in range(n_samples):
            cls_index = np.where(classes == y[i])[0][0]
            counts[cls_index] += 1

        max_samples = np.max(counts)
        X_by_class = np.zeros((n_classes, max_samples, n_features))

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        row_trackers = np.zeros(n_classes, dtype=int)

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

        self.covariance_ = np.zeros((n_classes, n_features, n_features))

        for i in range(n_classes):
            self.covariance_[i] = (
                (X_by_class[i][:counts[i]] - self.means_[i]).T @ 
                (X_by_class[i][:counts[i]] - self.means_[i]) / counts[i]
            )

        self.inv_cov_ = np.zeros((n_classes, n_features, n_features))
        self.log_det_ = np.zeros(n_classes)

        reg = 1e-6
        for i in range(n_classes):
            cov_reg = self.covariance_[i] + reg * np.eye(n_features)
            self.inv_cov_[i] = np.linalg.inv(cov_reg)
            self.log_det_[i] = np.linalg.det(cov_reg)

        self.score_ = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            self.score_[:, i] = (
                -0.5 * np.sum((X @ self.inv_cov_[i]) * X, axis=1) +
                X @ self.inv_cov_[i] @ self.means_[i] -
                0.5 * self.means_[i].T @ self.inv_cov_[i] @ self.means_[i] -
                0.5 * np.log(self.log_det_[i]) + np.log(self.prior_[i])
            )

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            quadratic = -0.5 * np.sum((X @ self.inv_cov_[i]) * X, axis=1)
            linear = X @ self.inv_cov_[i] @ self.means_[i]
            intercept = (
                -0.5 * self.means_[i].T @ self.inv_cov_[i] @ self.means_[i] -
                0.5 * np.log(self.log_det_[i]) + np.log(self.prior_[i])
            )

            scores[:, i] = quadratic + linear + intercept
        
        return self.classes_[np.argmax(scores, axis=1)]

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

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

train_pred = qda.predict(X_train)
test_pred = qda.predict(X_test)

train_acc = np.mean(train_pred == y_train)
test_acc = np.mean(test_pred == y_test)

print(f"Custom QDA - Train accuracy: {train_acc:.4f}")
print(f"Custom QDA - Test accuracy: {test_acc:.4f}")

sk_qda = SKQDA(reg_param=0.0)
sk_qda.fit(X_train, y_train)
sk_test_pred = sk_qda.predict(X_test)
sk_test_acc = np.mean(sk_test_pred == y_test)

print(f"sklearn QDA - Test accuracy: {sk_test_acc:.4f}")
print(f"Match? {np.all(test_pred == sk_test_pred)}")