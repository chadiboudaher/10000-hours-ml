# Linear Model Selection and Regularization

Why might we want to use another fitting procedure instead of least squares? alternative fitting procedures can yield better _prediction accuracy_ and _model interpretability_

- _Prediction Accuracy_: Given that the relation between the response and the predictors is approximately linear, the least squares estimates will have low bias. if $n >> p$ then the least squares estimates tend to also have low variance, and hence will perform well on test observations. However, if n is not much larger than p, then there can be a lot of variability in the least squares fit, resulting in overfitting and consequently poor predictions on future observations not used in model training.

- _Model Interpretability_: It is often the case that some or many of the variables used in a multiple regression model are in fact not associated with the response.

There are many method can be used to fit the model:

- **Subset Selection**: his approach involves identifying a subset of the predictors that we believe to be related to the response. We then fit a model using least squares on the reduced set of variables.

- **Shrinkage**: This approach involves fitting a model involving all p predictors. However, the estimated coefficients are shrunken towards zero relative to the least squares estimates. This shrinkage (also known as _regularization_) has the effect of reducing variance. Depending on what type of shrinkage is performed, some of the coefficients may be estimated to be exactly zero. Hence, shrinkage methods can also perform variable selection.

- **Dimension Reduction**: This approach involves projecting the p predictors into an M -dimensional subspace, where $M < p$.
