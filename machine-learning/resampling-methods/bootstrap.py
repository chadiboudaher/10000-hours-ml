import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([65, 70, 75, 85, 90])

n_bootstraps = 1000
estimates = []

for _ in range(n_bootstraps):
    idx = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X[idx]
    y_boot = y[idx]

    model = LinearRegression().fit(X_boot, y_boot)
    estimates.append(model.coef_[0])

se_bootstrap = np.std(estimates)
print(f"Bootstrap SE: {se_bootstrap:.3f}")