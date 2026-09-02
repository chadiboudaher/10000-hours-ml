# Forward Stepwise Selection — Diabetes Dataset

This experiment implements **Forward Stepwise Selection** from scratch using the `load_diabetes` regression dataset from Scikit-learn.

The goal is to understand how greedy stepwise feature selection works, evaluate the sequence of selected models using several model-selection criteria, and compare its behavior with exhaustive Best Subset Selection.

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

The data was divided into:

```text
Training set: 353 observations
Test set:      89 observations
```

The predictors are numeric and already standardized.

---

## Forward Stepwise Selection

Forward Stepwise Selection starts with the **null model**, containing no predictors.

At each step:

1. Keep all predictors already selected.
2. Add each remaining predictor one at a time.
3. Fit a linear regression model for every candidate.
4. Calculate the training RSS.
5. Select the predictor that produces the lowest RSS.
6. Permanently add that predictor to the model.
7. Repeat until all predictors have entered.

The important property of forward selection is that once a predictor enters the model, it cannot later be removed.

---

## Algorithm

Let \(M_0\) denote the null model.

For

$$
k = 0,1,\ldots,p-1
$$

consider all models obtained by adding one unused predictor to \(M_k\).

From those \(p-k\) candidate models, select the one with the smallest:

$$
RSS=\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

and call the resulting model:

$$
M_{k+1}
$$

This produces a nested sequence:

$$
M_0 \subset M_1 \subset M_2 \subset \cdots \subset M_p
$$

Unlike Best Subset Selection, Forward Stepwise Selection does not reconsider every possible subset for each model size.

---

## Selected Models

The forward-selection sequence obtained on the training set was:

|   k | Selected Features                      | Training RSS |
| --: | -------------------------------------- | -----------: |
|   1 | `bmi`                                  | 1,360,501.77 |
|   2 | `bmi, s5`                              | 1,161,834.75 |
|   3 | `bmi, s5, bp`                          | 1,108,649.50 |
|   4 | `bmi, s5, bp, s1`                      | 1,074,425.53 |
|   5 | `bmi, s5, bp, s1, s2`                  | 1,053,574.06 |
|   6 | `bmi, s5, bp, s1, s2, sex`             | 1,022,172.26 |
|   7 | `bmi, s5, bp, s1, s2, sex, s4`         | 1,016,695.10 |
|   8 | `bmi, s5, bp, s1, s2, sex, s4, s6`     | 1,015,027.32 |
|   9 | `bmi, s5, bp, s1, s2, sex, s4, s6, s3` | 1,013,490.08 |
|  10 | All predictors                         | 1,012,598.05 |

As expected, training RSS decreases as more predictors are added.

---

## Comparison with Best Subset Selection

Forward Stepwise Selection matched the Best Subset model for most model sizes.

However, an important difference appeared at:

$$
k=5
$$

Best Subset Selection found:

```text
sex, bmi, bp, s3, s5
```

with:

$$
RSS=1,040,983.57
$$

Forward Stepwise Selection found:

```text
bmi, s5, bp, s1, s2
```

with:

$$
RSS=1,053,574.06
$$

Therefore:

$$
RSS_{\text{Best Subset}} < RSS_{\text{Forward}}
$$

This happens because Forward Stepwise Selection had already selected `s1` at the previous step and was not allowed to remove it.

This illustrates the main limitation of the greedy forward-selection procedure:

> Forward Stepwise Selection is computationally cheaper than exhaustive Best Subset Selection, but it is not guaranteed to find the globally optimal subset for every model size.

Interestingly, from \(k=6\) onward, the Forward Stepwise models again matched the Best Subset models.

---

## Adjusted \(R^2\)

Adjusted \(R^2\) was calculated for every model in the forward-selection sequence:

$$
R^2_{adj}
=
1-(1-R^2)
\frac{n-1}{n-k-1}
$$

Results:

|     k | Adjusted \(R^2\) |
| ----: | ---------------: |
|     1 |           0.3639 |
|     2 |           0.4552 |
|     3 |           0.4787 |
|     4 |           0.4933 |
|     5 |           0.5017 |
|     6 |           0.5152 |
| **7** |       **0.5164** |
|     8 |           0.5158 |
|     9 |           0.5151 |
|    10 |           0.5141 |

The maximum adjusted \(R^2\) occurred at:

$$
\boxed{k=7}
$$

with the model:

```text
bmi, s5, bp, s1, s2, sex, s4
```

After seven predictors, the improvement in training fit was no longer sufficient to compensate for the additional model complexity.

---

## Bayesian Information Criterion

BIC was also used for model selection.

For linear regression:

$$
BIC
=
n\ln\left(\frac{RSS}{n}\right)
+
q\ln(n)
$$

where:

- \(n\) is the number of training observations
- \(RSS\) is the residual sum of squares
- \(q\) is the number of estimated parameters, including the intercept

Lower BIC values are preferred.

### Results

|     k |           BIC |
| ----: | ------------: |
|     1 |     2926.4173 |
|     2 |     2876.5616 |
|     3 |     2865.8872 |
|     4 |     2860.6848 |
|     5 |     2859.6333 |
| **6** | **2854.8186** |
|     7 |     2858.7885 |
|     8 |     2864.0754 |
|     9 |     2869.4068 |
|    10 |     2874.9625 |

The minimum BIC occurred at:

$$
\boxed{k=6}
$$

with:

```text
bmi, s5, bp, s1, s2, sex
```

Thus, BIC preferred a smaller model than adjusted \(R^2\).

---

## Cross-Validation

A **5-fold cross-validation** procedure was performed on the training set.

To avoid information leakage, Forward Stepwise Selection was repeated independently inside every training fold.

For every candidate model size \(k\):

1. Split the training set into five folds.
2. Use four folds as the fold-training data.
3. Perform Forward Stepwise Selection using only that data.
4. Select exactly \(k\) predictors.
5. Fit a linear regression model.
6. Evaluate it on the validation fold.
7. Repeat across all five folds.
8. Average the validation MSE.

The optimal model size was:

$$
\boxed{k=6}
$$

with average CV MSE:

$$
\boxed{3024.93}
$$

Cross-validation therefore agreed with BIC.

---

## Final Model

The CV-selected model size was:

$$
k=6
$$

Running Forward Stepwise Selection on the complete training set selected:

```text
bmi, s5, bp, s1, s2, sex
```

The final model was trained using these six predictors and evaluated on the untouched test set.

### Test Results

| Metric       |       Value |
| ------------ | ----------: |
| Test MSE     | **2846.29** |
| Test RMSE    |   **53.35** |
| Test \(R^2\) |  **0.4628** |

---

## Model Selection Summary

| Criterion               | Selected Model Size |
| ----------------------- | ------------------: |
| Training RSS            |                  10 |
| Adjusted \(R^2\)        |                   7 |
| BIC                     |               **6** |
| 5-Fold Cross-Validation |               **6** |

Both BIC and cross-validation selected a six-predictor model.

---

## Best Subset vs Forward Stepwise

| Property                                | Best Subset Selection | Forward Stepwise Selection |
| --------------------------------------- | --------------------- | -------------------------- |
| Search strategy                         | Exhaustive            | Greedy                     |
| Reconsiders previous choices            | Yes                   | No                         |
| Models nested                           | Not necessarily       | Yes                        |
| Guaranteed best subset for each \(k\)   | Yes                   | No                         |
| Computational cost                      | High                  | Lower                      |
| Can remove previously selected features | Yes                   | No                         |

For \(p\) predictors, Best Subset Selection must consider:

$$
2^p
$$

possible subsets.

Forward Stepwise Selection considers approximately:

$$
p+(p-1)+(p-2)+\cdots+1
$$

candidate additions, which equals:

$$
\frac{p(p+1)}{2}
$$

For the Diabetes dataset with \(p=10\):

$$
\frac{10(11)}{2}=55
$$

candidate models are considered during Forward Stepwise Selection, compared with up to:

$$
2^{10}=1024
$$

possible subsets in exhaustive subset search.

---

## Key Observations

- Forward Stepwise Selection builds models incrementally.
- Once a predictor enters the model, it cannot be removed.
- The resulting models are nested.
- Training RSS decreases as predictors are added.
- Forward Stepwise Selection can miss the globally best model because early greedy choices restrict later models.
- This limitation was directly observed at \(k=5\).
- Adjusted \(R^2\) selected seven predictors.
- BIC selected six predictors.
- Five-fold cross-validation also selected six predictors.
- The final six-predictor model achieved a test MSE of 2846.29 and test \(R^2\) of approximately 0.463.
- Forward Stepwise Selection evaluated substantially fewer candidate models than exhaustive Best Subset Selection.

---

## Tools

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Main Scikit-learn components:

```python
LinearRegression
KFold
mean_squared_error
r2_score
```

---

## Conclusion

Forward Stepwise Selection provides a computationally efficient alternative to exhaustive Best Subset Selection.

Instead of considering every possible predictor combination, it constructs a nested sequence of models by greedily adding the predictor that produces the largest improvement at each step.

On the Diabetes dataset, the method generally produced the same models as Best Subset Selection, although it failed to identify the globally optimal five-feature model because of an earlier greedy decision.

Adjusted \(R^2\) preferred a seven-feature model, while both BIC and 5-fold cross-validation selected six predictors.

The final six-predictor model achieved:

$$
MSE=2846.29
$$

$$
RMSE=53.35
$$

$$
R^2=0.4628
$$

on the held-out test set.

The experiment demonstrates the main tradeoff of Forward Stepwise Selection:

> It sacrifices the guarantee of finding the globally best subset in exchange for substantially lower computational cost.

The next natural extension is **Backward Stepwise Selection**, which begins with the full model and removes predictors one at a time.
