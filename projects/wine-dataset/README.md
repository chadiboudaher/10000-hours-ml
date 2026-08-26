# Classification Algorithm Comparison on Wine Dataset

## 1. Overview

This project compares four popular classification algorithms on the UCI Wine dataset:

- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Logistic Regression
- Gaussian Naive Bayes

The goal is to evaluate and compare their performance in classifying wine varieties based on chemical features.

## 2. Dataset

Samples: 178
Features: 13
Classes: 3

## 3. Methodology

### 3.1 Preprocessing

- Train-test split: 70/30 with stratification
- Feature scaling using StandardScaler
- No missing values or categorical features to encode

### 3.2 Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: Weighted average
- **Recall**: Weighted average
- **F1-Score**: Harmonic mean of precision and recall
- **Training & Prediction Time**: Computational efficiency
- **Confusion Matrices**: Class-wise performance

## 4. Results

### 4.1 Summary

| Model               | Accuracy | Precision | Recall | F1-Score | Train Time (s) | Pred Time (s) |
| ------------------- | -------- | --------- | ------ | -------- | -------------- | ------------- |
| LDA                 | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 0.0019         | 0.0003        |
| Logistic Regression | 1.0000   | 1.0000    | 1.0000 | 1.0000   | 0.0138         | 0.0005        |
| QDA                 | 0.9722   | 0.9741    | 0.9722 | 0.9718   | 0.0012         | 0.0003        |
| Naive Bayes         | 0.9722   | 0.9741    | 0.9722 | 0.9722   | 0.0012         | 0.0003        |

## 5. Key Findings

1. LDA & Logistic Regression achieved perfect classification (100% accuracy)
   - Wine dataset features satisfy LDA's assumptions well
   - Classes are linearly separable

2. QDA & Naive Bayes tied with 97.22% accuracy
   - Flexibility of QDA didn't provide advantage
   - Naive Bayes' independence assumption didn't significantly hurt performance

3. All models are computationally efficient
   - Fastest: QDA and Naive Bayes (0.0012s training)
   - All models predict in < 0.001 seconds

## 6. Requirements

- numpy
- pandas
- scikit-learn
- matplotlib

## 7. Key Takeaways

- LDA is the best choice for this dataset: fast, accurate, and interpretable
- Logistic Regression is equally accurate but slightly slower to train

## 8. References

1. [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). _An Introduction to Statistical Learning with Python_. Springer.
