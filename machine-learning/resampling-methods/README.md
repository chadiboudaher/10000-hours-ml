# Resampling Methods

## 1. The Validation Set Approach

The validation set is the simplest method for model selection and hyperparameter tuning. It split the data into two parts: a `training set` and a `validation set`. We train the model on the training set and evaluate its performence on the validation set.

### 1.1 Process

1. **Split the data**: A typical split is **70%** for training and **30%** for validation.
2. **Train**: Use model to train on the training set.
3. **Validate**: Use trained model to make prediction on the validation set.
4. **Evaluate**: Calculate a performence metric on the validation set.
5. **Select**: Compare the validation scores of different models or hyperparameters, and choose the one with the best score.

### 1.2 Is it the same as Test set?

The validation set is not the same as the test set. If we use the validation set to pick our model, and then report that validation accuracy as our "final" accuracy, our result will be biased.

### 1.3 Major Disadvantages

- **High Variance**: The validation score is highly dependent on which specific data points end up in the validation set.
- **Reduced Training data**: Bad for small datasets. It is prefferably used on large datasets.
- Unstable model selection: Because of high variance, if we repeat the random split multiple times, we might end up choosing a completely different "best" model each time.
