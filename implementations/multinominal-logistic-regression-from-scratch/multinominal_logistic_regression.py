import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42

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
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

np.random.seed(RANDOM_SEED)

X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_classes=4,
    n_informative=4,
    n_redundant=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    flip_y=0.0,
    random_state=42
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True
    )
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

my_model = MultinominalLogisticRegression(
    n_features=8,
    n_classes=4
)
my_model.fit(X_train, y_train, epochs=800, lr=0.1)

sk_model = LogisticRegression(max_iter=800)
sk_model.fit(X_train, y_train)


y_pred = my_model.predict(X_test)
predictions = my_model.predict_prob(X_test)

y_pred_sk = sk_model.predict(X_test)
predictions_sk = sk_model.predict_proba(X_test)

accuracy = np.mean(y_pred == y_test)
accuracy_sk = np.mean(y_pred_sk == y_test)
print(f"Test Accuracy: {accuracy}")
print(f"Test Accuracy SKLEARN: {accuracy_sk}")

OHE = OneHotEncoder(sparse_output=False)
y_test_onehot = OHE.fit_transform(y_test.reshape(-1, 1))
loss = compute_loss(y_test_onehot, predictions)
loss_sk = compute_loss(y_test_onehot, predictions_sk)
print(f"Test Loss: {loss}")
print(f"Test Loss SKLEARN: {loss_sk}")

my_model.plot_loss()