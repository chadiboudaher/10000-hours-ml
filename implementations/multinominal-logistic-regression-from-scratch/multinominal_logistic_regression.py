import numpy as np

def softmax(z):
    """
    A problem is introduced when implementing softmax activation function.
    This problem is called overflow, as the number of the exponent of exp is
    greater the result is greater, creating very large number, which result
    in overflowing the (finite) memory allocated for the variable. So we
    have to remove the overflow, but how?

    We solve this issue using the "Exp-normalize trick".
    """
    maximum = np.max(z, axis=-1, keepdims=True)
    e_x = np.exp(z - maximum)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def compute_loss(y, p):
    """
    Categorical cross entropy is used with multi-class classification
    problems.
    """
    m = len(y)
    eps = 1e-15

    p = np.clip(p, eps, 1-eps)
    L = (-1/m) * np.sum(y * np.log(p))
    return L

def compute_gradients(X, y_onehot, p):
    m = X.shape[0]

    dL_dz = p - y_onehot
    dw = (1/m) * X.T @ dL_dz
    db = (1/m) * np.sum(dL_dz, axis=0)
    
    return dw, db

class MultinominalLogisticRegression:
    def __init__(self, n_features, n_classes):
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.loss_history = []

    def fit(self, X, y, epochs=1000, lr=0.1):
        n_classes = self.weights.shape[1]
        y_onehot = np.eye(n_classes)[y]

        for epoch in range(epochs):
            z = X @ self.weights + self.bias
            p = softmax(z)

            dw, db = compute_gradients(X, y_onehot, p)

            self.weights -= lr * dw
            self.bias -= lr * db
            
            loss = compute_loss(y_onehot, p)
            self.loss_history.append(loss)
            
            if epoch % 100 == 0:
                predictions = np.argmax(p, axis=1)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}, loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict_prob(self, X):
        """Return predicted probabilities"""
        z = X @ self.weights + self.bias
        return softmax(z)

    def predict(self, X):
        """Return predicted class labels"""
        probs = self.predict_prob(X)
        return np.argmax(probs, axis=1)

    def plot_loss(self):
        """Plot training loss history"""
        import matplotlib.pyplot as plt
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()