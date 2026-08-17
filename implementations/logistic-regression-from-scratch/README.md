# Logistic Regression From Scratch

A simple Logistic Regression model implemented from scratch using NumPy.

## Features

- Sigmoid activation
- Binary cross-entropy loss
- Gradient computation
- Gradient descent
- Probability prediction
- Binary classification with a custom threshold
- Training loss visualization

## Model

The model computes:

[
z = Xw + b
]

and converts the result into a probability using the sigmoid function:

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

The weights and bias are updated using gradient descent.

## Testing

The model was tested on synthetic classification data generated with `scikit-learn`.

The dataset was:

- Split into training and testing sets
- Standardized using `StandardScaler`
- Evaluated on unseen test data
- Compared with `sklearn.linear_model.LogisticRegression`

## Usage

```python
model = LogisticRegression(numOfFeatures=5)

model.fit(
    X_train,
    y_train,
    epochs=500,
    lr=0.1
)

predictions = model.predict(X_test)

model.plot_loss()
```

## Libraries

- NumPy
- Matplotlib
- scikit-learn

## Goal

The purpose of this implementation is to understand how Logistic Regression works internally instead of only using a library implementation.
