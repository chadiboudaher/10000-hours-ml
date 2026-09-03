# Backward Stepwise Selection — Diabetes Dataset

This experiment implements **Backward Stepwise Selection** from scratch using Scikit-learn's `load_diabetes` dataset.

## Dataset

- 442 observations
- 10 predictors
- 353 training samples
- 89 test samples

Predictors:

```text
age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
```

## Method

Backward Stepwise Selection starts with all predictors.

At each step:

1. Remove each predictor one at a time.
2. Fit all candidate models.
3. Calculate RSS.
4. Keep the model with the lowest RSS.
5. Repeat until only one predictor remains.

Unlike Best Subset Selection, previously removed predictors cannot re-enter the model.

## Selected Models

|   k | Features                             |          RSS | Adjusted R² |
| --: | ------------------------------------ | -----------: | ----------: |
|   9 | sex, bmi, bp, s1, s2, s3, s4, s5, s6 | 1,013,490.08 |      0.5151 |
|   8 | sex, bmi, bp, s1, s2, s4, s5, s6     | 1,015,027.32 |      0.5158 |
|   7 | sex, bmi, bp, s1, s2, s4, s5         | 1,016,695.10 |  **0.5164** |
|   6 | sex, bmi, bp, s1, s2, s5             | 1,022,172.26 |      0.5152 |
|   5 | bmi, bp, s1, s2, s5                  | 1,053,574.06 |      0.5017 |
|   4 | bmi, bp, s1, s5                      | 1,074,425.53 |      0.4933 |
|   3 | bmi, bp, s5                          | 1,108,649.50 |      0.4787 |
|   2 | bmi, s5                              | 1,161,834.75 |      0.4552 |
|   1 | bmi                                  | 1,360,501.77 |      0.3639 |

Adjusted \(R^2\) selected:

$$
\boxed{k=7}
$$

## BIC

BIC was calculated as:

$$
BIC=n\ln(RSS/n)+q\ln(n)
$$

where \(q\) includes the model parameters.

The minimum BIC occurred at:

$$
\boxed{k=6}
$$

with:

```text
sex, bmi, bp, s1, s2, s5
```

and:

$$
BIC=2854.8186
$$

## Comparison

| Method            | Adjusted R² | BIC |
| ----------------- | ----------: | --: |
| Best Subset       |           7 |   6 |
| Forward Stepwise  |           7 |   6 |
| Backward Stepwise |           7 |   6 |

At \(k=5\), Backward Stepwise did not find the globally best subset found by Best Subset Selection. This illustrates that stepwise methods are greedy and are not guaranteed to find the optimal subset for every model size.

## Conclusion

Backward Stepwise Selection provides a computationally cheaper alternative to exhaustive Best Subset Selection by progressively removing predictors.

For this experiment:

- Adjusted \(R^2\) selected 7 predictors.
- BIC selected 6 predictors.
- The 6-feature model matched the model selected by Best Subset and Forward Stepwise Selection.

The next step is to compare all three subset-selection methods before moving to **Ridge Regression**.
