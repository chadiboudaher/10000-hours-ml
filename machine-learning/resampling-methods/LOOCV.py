import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    random_state=RANDOM_SEED
)

loo = LeaveOneOut()

y_true = []
y_pred = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestClassifier(
        n_estimators=50, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    y_pred.append(model.predict(X_test)[0])
    y_true.append(y_test[0])

loocv_accuracy = accuracy_score(y_true, y_pred)
print(f"LOOCV Accuracy: {loocv_accuracy:.4f}")
print(f"Number of models trained: {len(y_true)}")