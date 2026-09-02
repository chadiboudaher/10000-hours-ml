# Linear Model Selection and Regularization

Why might we want to use another fitting procedure instead of least squares? alternative fitting procedures can yield better _prediction accuracy_ and _model interpretability_

- _Prediction Accuracy_: Given that the relation between the response and the predictors is approximately linear, the least squares estimates will have low bias. if n >> p then the least squares estimates tend to also have low variance, and hence will perform well on test observations. However, if n is not much larger than p, then there can be a lot of variability in the least squares fit, resulting in overfitting and consequently poor predictions on future observations not used in model training.

- _Model Interpretability_: It is often the case that some or many of the variables used in a multiple regression model are in fact not associated with the response.

There are many method can be used to fit the model:

- **Subset Selection**: his approach involves identifying a subset of the predictors that we believe to be related to the response. We then fit a model using least squares on the reduced set of variables.

- **Shrinkage**: This approach involves fitting a model involving all p predictors. However, the estimated coefficients are shrunken towards zero relative to the least squares estimates. This shrinkage (also known as _regularization_) has the effect of reducing variance. Depending on what type of shrinkage is performed, some of the coefficients may be estimated to be exactly zero. Hence, shrinkage methods can also perform variable selection.

- **Dimension Reduction**: This approach involves projecting the p predictors into an M -dimensional subspace, where M < p.

## Subset Selection

### Best Subset Selection

To perform _best subset selection_, we fit a seperate squares regression for each possible combination of the p predictors. We then look at all of the resulting models, with the goal of identifying the one that is **best**.

The problem of selecting the best model from among the 2^P possibilities considered by best subset selection is not trivial.
and it is usually broken up into 2 stages.

#### Algorithm - Best subset Selection

1. Let $$ M_0 $$ denote the null model, which contains no predictors. This model simply predicts the sample mean for each observation.
2. For k = 1,2, ...p:
   - Fit all models that contains k predictors.
   - Pick the best among models. Here **best** is defined as having smallest _RSS_ or _deviance_ in **Logistic Regression**.
3. Select a single best model from among using the prediction error on a validation set.

But with the increase in number of p, the number of possibilities increase, increasing the computational complexity. There are computational shortcuts, so called branch and bound techniques for eliminating some choices. but these have their limitations as p gets large. They also only work for least squares linear regression. We present computationally efficient alternatives to best subset selection next.

## Stepwise Selection

For computational reasons, best subset selection cannot be applied with very large p. It may also struggle from statistical problems when p is large.

**Stepwise selection**, explore a far more restricted set of models, are attractive alternatives to best subset selection.

### Forward Stepwise selection

_Forward stepwise selection_ is a computationally efficient alternative to best subset selection.

Forward stepwise selection begins with a model containing no predictors, and then adds predictors to the model, one-at-a-time, until all of the predictors are in the model. In particular, at each step the variable that gives the greatest additional
improvement to the fit is added to the model. It fit $1+\frac{p(p+1)}{2}$ models.
