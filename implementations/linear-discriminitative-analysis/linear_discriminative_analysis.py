import numpy as np

RANDOM_SEED = 42

N_FEATURES = 4
N_SAMPLES = 100
N_CLASSES = 2
NOISE = 0.3

np.random.seed(RANDOM_SEED)

# Simulate data for LDA

X = np.random.rand(N_SAMPLES, N_FEATURES)
y = np.random.randint(0, N_CLASSES, N_SAMPLES)

for i in range(N_SAMPLES):
    if y[i] == 0:
        X[i] = np.random.uniform(0.3, 0.5, size=N_FEATURES)
    else:
        X[i] = np.random.uniform(0.8, 0.3, size=N_FEATURES)
# print(X)
# print(y)

mean = np.zeros((N_CLASSES, N_FEATURES))
print(mean)

class LinearDiscriminantAnalysis:
    def __init__(self, n_samples=100,n_features=4, n_classes=2):
        self.means = None
        self.priors = None
        self.scatter = None
        self.classes = None

    def fit(self, X, y):
        """
        means: Shape, we have one for each class. (n_classes, n_features)
        prior: Shape, we have one for each class. (n_classes,)
        scatter: variance (scalar)
        classes: unique classes labels
        """

        # Get classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        n_samples, n_features = X.shape
        self.means = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        total_scatter = 0

        for idx, class_label in enumerate(self.classes):
            X_k = X[y == class_label]
            n_k = X_k.shape[0]

            self.means[idx] = X_k.mean(axis=0)

            self.priors[idx] = n_k / n_samples

            deviations = X_k - self.means[idx]
            scatter_k = np.sum(deviations**2)
            total_scatter += scatter_k

        self.scatter = total_scatter / (n_samples - n_classes)

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        deltas = np.zeros((n_samples, n_classes))
        for idx, class_label in enumerate(self.classes):
            mu_k = self.means[idx]
        
            dot_product = X @ mu_k
            quadratic = np.sum(mu_k**2) / (2 * self.scatter)
            prior_term = np.log(self.priors[idx])
            
            deltas[:, idx] = dot_product / self.scatter - quadratic + prior_term
    
        return self.classes[np.argmax(deltas, axis=1)]

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        deltas = np.zeros((n_samples, n_classes))
        
        for idx, class_label in enumerate(self.classes):
            mu_k = self.means[idx]
            dot_product = X @ mu_k
            quadratic = np.sum(mu_k**2) / (2 * self.scatter)
            deltas[:, idx] = dot_product / self.scatter - quadratic + np.log(self.priors[idx])

        deltas_shifted = deltas - np.max(deltas, axis=1, keepdims=True)
        exp_deltas = np.exp(deltas_shifted)
        probabilities = exp_deltas / np.sum(exp_deltas, axis=1, keepdims=True)
        
        return probabilities

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

y_pred = lda.predict(X)

y_proba = lda.predict_proba(X)

accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.3f}")