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

## 2. Leave-One-Out Cross-Validation

`LOOCV` is an extreme form of k-fold cross-validation where k = n (The number of sample in your dataset). For each iteration, we train on all samples except one and validate on the single left-out sample. we repeat this process n times, once for each sample, and average the results.

### 2.1 Advantages of LOOCV

- **No randomness**: Results are deterministic (same every time).
- **Maximum training data**: Almost all data is being used except one samples.
- **Best for small datasets**.
- **Unbiased estimate**.

### 2.2 Disadvantages of LOOCV

- **Computationally expensive**: train n models.
- **high variance**.
- **Not for large datasets**.
- **no parallelization**.

## 3. K-Fold Cross-Validation

**K-Fold Cross-Validation** is a resampling method that splits your dataset into **k equal-sized folds**. You train your model on **k-1 folds** and validate on the 1 remaining fold, repeating this process k times.

_k_: Number of groups that a given data sample is to be split into.

### 3.1 Procedure

1. Shuffle the dataset randomly.
2. Split the dataset into k groups.
3. For each unique group:
   1. Take the group as a hold out or test data set
   2. Take the remaining groups as a training data set
   3. Fit a model on the training set and evaluate it on the test set
   4. Retain the evaluation score and discard the model
4. Summarize the skill of the model using the sample of model evaluation scores

### Configuration of K

The k value must be chosen carefully for your data sample.

Three common tactics for choosing a value for k are as follows:

1. **K=10**: The value for k is fixed to 10, a value that has been found through experimentation to generally result in a model skill estimate with low bias a modest variance.
2. LOOCV

## 4. Bias-Variance Trade-Off for k-Fold Cross-validation

Putting computational issues aside, a less obvious but potentially more important advantage of k-fold CV is that it often gives more accuracte estimates of the test error rate than does LOOCV. This has to do with a bias-variance trade-off.

in LOOCV, training is done on n-1 observations, which results in an unbiased estimates of the test error. Unlike k-fold CV, which lead to an intermediate level of bias, since each training set contains $\frac{(k-1)n}{k}$ observations.

Therefore, from the perspective of bias reduction, it is clear that LOOCV is to be preferred to k-fold CV, But the test error estimate resulting from LOOCV tends to have higher variance.

## 5. Cross-Validation on Classification Problems

In classification, adding polynomial terms can improve test performance up to a point, but too much flexibility (higher-degree polynomials) leads to overfitting, and cross-validation helps you find that sweet spot.
