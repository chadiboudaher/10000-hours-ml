import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, p):
    m = len(y)
    eps = 1e-15
    p = np.clip(p, eps, 1-eps) # Prevent log(0)
    L = (-1 / m) * (y @ np.log(p) + (1 - y) @ np.log(1 - p))
    return L

def compute_gradients(X, y, p):
    m = len(y)

    p = p.reshape(-1, 1)
    y = y.reshape(-1, 1)

    dw = (-1 / m) * np.sum(y * (X * (1 - p)) + (1 - y) * (-p * X), axis=0)
    db = (-1 / m) * np.sum(y * (1 - p) + (1 - y) * (-p))

    # print(dw.shape)
    # print(db.shape)

    return dw, db

class LogisticRegression:
    """
    Args:
        numOfFeatures: the number of inputs features.
    """
    def __init__(self, numOfFeatures):
        self.weights = np.zeros(numOfFeatures)
        self.bias = 0 

    def fit(self, X, y, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            # Make a prediction
            z = X @ self.weights + self.bias

            # Apply logistic function
            p = sigmoid(z)

            # Compute gradients
            dw, db = compute_gradients(X, y, p)

            # Update parameters
            self.weights -= lr * dw
            self.bias -= lr * db

            if epoch % 100 == 0:
                loss = compute_loss(y, p)
                print(f"Epoch {epoch}, loss: {loss:.4f}")

    def predict_prob(self, X):
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int)

# Example for compute gradients
# X = np.random.randn(100, 3)
# y = np.random.randint(0, 2, 100)
# p = np.random.rand(100)

# compute_gradients(X, y, p)

# Test with synthetic data

# X = np.random.randn(1000, 3)
# true_weights = np.array([2, -1, 0.5])
# true_bias = -0.5
# z = X @ true_weights + true_bias
# y = (sigmoid(z) > 0.5).astype(int)

# model = LogisticRegression(numOfFeatures=3)
# model.fit(X, y, epochs=1500, lr=0.01)

# predictions = model.predict(X)
# accuracy = np.mean(predictions == y)

# print(f"\nAccuracy: {accuracy:.4f}")
# print(f"True weights: {true_weights}")
# print(f"Learned weight: {model.weights}")
# print(f"True bias: {true_bias}")
# print(f"Learned bias: {model.bias:.4f}")

X, y = make_classification(random_state=42,
                           n_samples=1000,
                           n_features=5,
                           n_classes=2,
                           n_clusters_per_class=1,
                           shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(numOfFeatures=5)
model.fit(X_train, y_train, epochs=1000, lr=0.01)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
loss = compute_loss(y_pred, y_test)
print(f"Loss: {loss:.4f}")