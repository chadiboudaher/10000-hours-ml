# Classification

## 1. Why not Linear Regression?

There are two reasons:

1. Regression method cannot accommodate a qulitative response with more than two classes.
2. A regression method will not provide meaningful estimates of $P_r(Y | X), even with just two classes.$

## 2. Logistic Regression

we use the logistic function:

$$
p(X) = \frac{e^{\beta_0+\beta_1X}}{1+ e^{\beta_0+\beta_1X}}
$$

this function gives outputs between 0 and 1 for all values of $X$.

- to estimate the regression coefficients we use maximum likelihood to fit a logistic regression model. The estimates $\hat{\beta_0}$ and $\hat{\beta_1}$ are chosen to maximize this likelihood function. By `likelihood` we mean we are trying to find the optimal value of mmean and standart deviation for a distribution given a bunch of measured measurements.
