import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create Synthetic data

N_SAMPLES = 1000
N_CLASSES = 7
N_FEATURES = 10

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=8,  # 8 features are useful
    n_redundant=2,    # 2 features are noise
    n_classes=N_CLASSES,
    random_state=RANDOM_SEED
)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
validation_accuracy = accuracy_score(y_val, y_pred)

print(f"Validation Accuracy: {validation_accuracy:.4f}")

final_model = RandomForestClassifier(n_estimators=100)
final_model.fit(X, y)