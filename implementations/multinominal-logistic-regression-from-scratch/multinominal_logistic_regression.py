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

class MultinominalLogisticRegression:
    def __init__(self, n_features, n_classes):
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.loss_history = []

    def fit(self, X, y, epochs=1000, lr=0.1):
        z = X @ self.weights + self.bias
        p = softmax(z)

