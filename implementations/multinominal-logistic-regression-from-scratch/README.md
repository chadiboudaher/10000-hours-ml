# Multinominal Logistic Regression from scratch

MLR is a similar case to logistic regression but the difference is with the number of classes that can be predicted.

## Features

- softmax activation function + overflow problem and how to solve it.
- loss function (Categorical cross entropy).
- Gradient Computation.
- Gradient Descent.
- On-hot encoding usage (multi-class).
- Training loss visulization.

## Model

- **softmax activation** is used to scale output values in the range of (0, 1).
- Categorical cross entropy function is used to calculate loss.
- The shape of bias is equivalent to the `(n_classes)`.
- The shape of weights is equivalent to `(n_features, n_classes)`
- _Weights_ and _Biases_ are changed using gradient descent (backpropagation).

## Testing

The model was tested on synthetic classification data generated with `scikit-learn`.

The dataset was:

- Split into training and testing sets
- Standardized using `StandardScaler`
- Evaluated on unseen test data
- Compared with `sklearn.linear_model.LogisticRegression`

## Libraries

- NumPy
- Matplotlib
- scikit-learn

## Goal

The purpose of this implementation is to understand how Logistic Regression works internally instead of only using a library implementation.
