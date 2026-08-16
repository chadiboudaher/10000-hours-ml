# Classification

## 1. Problem / Motivation

### What problem is this method trying to solve?

Classification method is used is problems where the predicted output $Y$ is qualitative.

Qualitative data example:

- SPAM / NOT SPAM.
- Disease ? NO Disease.
- Dog / cat / bird.
- and many more...

### Why do we need it?

We need `classification` methods because simpler methods like `linear regression` does it work well with qualitative data (more precisely multinominal qualitative data)

## 2. Core Idea

Classification methods classifies data into classes base on its features or the data sample input $X$.

## 3. Mathematical Formulation

### 3.1 The logistic function

This function better represent the probabilities of the model by suppressing then into a range of (0, 1), disallowing negative and large value from accoring.

$$ P(x) = \frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}} $$

$\beta_0$: Intercept of the model.
$\beta_1$: Represent the a value that gives that best results based on the tained dataset.
$X$: Input feature.

### 3.2 The maximum likelihood

The goal is to find the optimal way to fit a distribution (normal, gamma, etc..) to the data. So instead of minimizing the distance of points from a line (like in linear regression), maximum likelihood asks: "what values of $\beta_0$ and $\beta_1$ make the observed data most probable?"

## How It Works

### 4.1 Discriminative Models (Logistic Regression)

It draws a boundary directly between the classes. it only cares about the line to seperate spam and not spam.

**Pros**: Usually performs well when you have a lot of data.
**Cons**: it becomes _unstable_ if classes are perfectly seperated.
