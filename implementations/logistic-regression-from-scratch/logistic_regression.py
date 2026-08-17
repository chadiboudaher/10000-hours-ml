import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, p):
    m = len(y)
    L = (-1 / m) * (y @ np.log(p) + (1 - y) @ np.log(1 - p))
    return L

def compute_gradients(X, y, p):
    m = len(y)

    p = p.reshape(-1, 1)
    y = y.reshape(-1, 1)

    dw = (-1 / m) * np.sum(y * (X * (1 - p)) + (1 - y) * (-p * X), axis=0)
    db = (-1 / m) * np.sum(y * (1 - p) + (1 - y) * (-p))

    print(dw.shape)
    print(db.shape)

    return dw, db

class LogisticRegression:
    """
    Args:
        numOfFeatures: the number of inputs features.
    """
    def __init__(self, numOfFeatures):
        self.weights = np.zeros(numOfFeatures)
        self.bias = 0 

    def fit(self, X, y, epochs=1000, lr=0.1):
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
X = np.random.randn(100, 3)
y = np.random.randint(0, 2, 100)
p = np.random.rand(100)

compute_gradients(X, y, p)