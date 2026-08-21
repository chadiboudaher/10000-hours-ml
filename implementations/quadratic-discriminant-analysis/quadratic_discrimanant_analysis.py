import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SKLDA


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

N_FEATURES = 7
N_CLASSES = 4
N_SAMPLES = 500

X = np.zeros((N_SAMPLES, N_FEATURES))
# print(X)

y = np.random.randint(0, N_CLASSES, N_SAMPLES)

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
        scores = self.score_
        
        return self.classes_[np.argmax(scores, axis=1)]

lda = QuadraticDiscriminantAnalysis()
lda.fit(X, y)

predictions = lda.predict(X)

accuracy = np.mean(predictions == y)
print(f"Training accuracy: {accuracy:.4f}")

sk_lda = SKLDA()
sk_lda.fit(X, y)
sk_preds = sk_lda.predict(X)
sk_accuracy = np.mean(sk_preds == y)
print(f"sklearn accuracy: {sk_accuracy:.4f}")
print(f"Match? {np.all(predictions == sk_preds)}")