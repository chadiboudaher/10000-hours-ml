# Best Subset Selection — Diabetes Dataset

This experiment implements **Best Subset Selection** from scratch using the `load_diabetes` regression dataset from Scikit-learn.

The goal is to understand how exhaustive feature subset selection works and compare different criteria for choosing the optimal model complexity.

## Dataset

The experiment uses:

```python
from sklearn.datasets import load_diabetes
```

The dataset contains:

- **442 observations**
- **10 predictor variables**
- **1 continuous target variable**

Predictors:

```text
age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
```

The dataset is already numeric, contains no missing values, and the predictors are standardized.

The data was divided into training and testing sets before performing model selection.

---

## Best Subset Selection

For \(p = 10\) predictors, Best Subset Selection evaluates every possible feature combination.

The total search space is:

$$
2^{10}=1024
$$

possible subsets.

For each model size

$$
k=1,2,\ldots,10
$$

all combinations containing exactly \(k\) predictors were fitted using ordinary least squares.

The subset producing the lowest training Residual Sum of Squares (RSS) was retained.

$$
RSS=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

---

## Best Subsets by Model Size

| Predictors | Best Subset                            | Training RSS |
| ---------: | -------------------------------------- | -----------: |
|          1 | `bmi`                                  | 1,360,501.77 |
|          2 | `bmi, s5`                              | 1,161,834.75 |
|          3 | `bmi, bp, s5`                          | 1,108,649.50 |
|          4 | `bmi, bp, s1, s5`                      | 1,074,425.53 |
|          5 | `sex, bmi, bp, s3, s5`                 | 1,040,983.57 |
|          6 | `sex, bmi, bp, s1, s2, s5`             | 1,022,172.26 |
|          7 | `sex, bmi, bp, s1, s2, s4, s5`         | 1,016,695.10 |
|          8 | `sex, bmi, bp, s1, s2, s4, s5, s6`     | 1,015,027.32 |
|          9 | `sex, bmi, bp, s1, s2, s3, s4, s5, s6` | 1,013,490.08 |
|         10 | All predictors                         | 1,012,598.05 |

As expected, training RSS continuously decreases as additional predictors are included.

This demonstrates why training RSS alone cannot be used to choose the optimal model: the full model will naturally achieve the lowest training RSS.

---

## Adjusted \(R^2\)

Adjusted \(R^2\) was used to account for increasing model complexity.

$$
R^2_{adj}
=
1-(1-R^2)
\frac{n-1}{n-k-1}
$$

Unlike ordinary \(R^2\), adjusted \(R^2\) can decrease when an additional predictor does not sufficiently improve the model.

### Results

|     k | Adjusted \(R^2\) |
| ----: | ---------------: |
|     1 |           0.3639 |
|     2 |           0.4552 |
|     3 |           0.4787 |
|     4 |           0.4933 |
|     5 |           0.5077 |
|     6 |           0.5152 |
| **7** |       **0.5164** |
|     8 |           0.5158 |
|     9 |           0.5151 |
|    10 |           0.5141 |

The maximum adjusted \(R^2\) occurred with:

$$
\boxed{k=7}
$$

Selected subset:

```text
sex, bmi, bp, s1, s2, s4, s5
```

---

## Bayesian Information Criterion

BIC was also used to select model complexity.

For linear regression, a comparison-equivalent form of BIC is:

$$
BIC
=
n\ln\left(\frac{RSS}{n}\right)
+
k\ln(n)
$$

where:

- \(n\) is the number of observations
- \(RSS\) is the residual sum of squares
- \(k\) represents the number of estimated model parameters

Lower BIC values are preferred.

### Results

|     k |           BIC |
| ----: | ------------: |
|     1 |     2926.4173 |
|     2 |     2876.5616 |
|     3 |     2865.8872 |
|     4 |     2860.6848 |
|     5 |     2855.3894 |
| **6** | **2854.8186** |
|     7 |     2858.7885 |
|     8 |     2864.0754 |
|     9 |     2869.4068 |
|    10 |     2874.9625 |

BIC selected:

$$
\boxed{k=6}
$$

with the predictors:

```text
sex, bmi, bp, s1, s2, s5
```

Although adding a seventh predictor reduced training RSS, the improvement was not sufficient to compensate for the additional model-complexity penalty.

---

## Held-Out Test Evaluation

Models selected using BIC and adjusted \(R^2\) were evaluated on the unseen test set.

The full 10-predictor model was also evaluated as a baseline.

| Model            | Predictors |    Test MSE | Test RMSE | Test \(R^2\) |
| ---------------- | ---------: | ----------: | --------: | -----------: |
| **BIC**          |      **6** | **2846.29** | **53.35** |   **0.4628** |
| Adjusted \(R^2\) |          7 |     2870.25 |     53.57 |       0.4583 |
| Full Model       |         10 |     2900.19 |     53.85 |       0.4526 |

The 6-predictor model selected by BIC achieved the best test performance among these models.

The full model had the smallest training RSS but the worst test performance of the three models, illustrating why minimizing training error alone does not necessarily lead to better generalization.

---

## Cross-Validation

To obtain a more reliable estimate of model performance, **5-fold cross-validation** was performed on the training set.

Feature selection was repeated independently inside every fold to avoid information leakage.

For each model size \(k\):

1. The fold's training portion was used to perform Best Subset Selection.
2. The best subset containing exactly \(k\) predictors was selected.
3. A linear regression model was fitted.
4. The model was evaluated on the fold's validation portion.
5. Validation MSE was averaged across all five folds.

The optimal model size was:

$$
\boxed{k=6}
$$

with an average cross-validation MSE of:

$$
\boxed{3037.99}
$$

This agrees with the model size selected using BIC.

---

## Final Comparison

| Selection Criterion     | Selected Model Size |
| ----------------------- | ------------------: |
| Training RSS            |                  10 |
| Adjusted \(R^2\)        |                   7 |
| BIC                     |               **6** |
| 5-Fold Cross-Validation |               **6** |

Both BIC and cross-validation preferred a 6-predictor model.

This experiment demonstrates that additional predictors can improve training fit without improving generalization.

---

## Key Observations

- Training RSS decreases monotonically as model complexity increases.
- The model with the lowest training RSS is not necessarily the model with the best test performance.
- Adjusted \(R^2\) penalizes unnecessary predictors and selected 7 features.
- BIC imposed a stronger complexity penalty and selected 6 features.
- Five-fold cross-validation also selected a model size of 6.
- The 6-feature BIC model achieved better test performance than both the 7-feature model and the full 10-feature model.
- Best subsets of different sizes are not necessarily nested. A feature appearing in the best model of size \(k\) can disappear from the best model of size \(k+1\).
- Best Subset Selection becomes computationally expensive as the number of predictors increases because the number of possible subsets grows exponentially.

---

## Tools

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- `itertools.combinations`

---

## Conclusion

Best Subset Selection provides an intuitive way to study the relationship between model complexity and predictive performance.

Using the Diabetes dataset, exhaustive search showed that the full 10-feature model achieved the lowest training RSS. However, model-selection criteria favored smaller models.

Adjusted \(R^2\) selected 7 predictors, while both BIC and 5-fold cross-validation selected 6 predictors. The 6-feature model also achieved the strongest performance on the held-out test set among the evaluated candidates.

The experiment highlights a central principle of statistical learning:

> Improving training fit does not necessarily improve performance on unseen data.

This provides motivation for more computationally efficient model-selection approaches such as **Forward Stepwise Selection** and **Backward Stepwise Selection**, as well as regularization techniques such as **Ridge Regression** and **Lasso**.
