# Classification

## 1. Why not Linear Regression?

There are two reasons:

1. Regression method cannot accommodate a qulitative response with more than two classes.
2. A regression method will not provide meaningful estimates of $P_r(Y | X), even with just two classes.$

## 2. Logistic Regression

we use the logistic function:

$$
p(X) = \frac{\exp{\beta_0+\beta_1X}}{1+ \exp{\beta_0+\beta_1X}}
$$
