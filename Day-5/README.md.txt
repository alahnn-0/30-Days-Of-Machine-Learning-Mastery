# Day 5 - Information Theory & Model Generalization

## Topics Covered

### 1. Sensitivity and Specificity

These metrics are commonly used for classification problems.

#### Sensitivity (True Positive Rate)

Sensitivity = TP / (TP + FN)

Measures the model's ability to correctly identify positive cases.

Example:
- Detecting patients with a disease.

---

#### Specificity (True Negative Rate)

Specificity = TN / (TN + FP)

Measures the model's ability to correctly identify negative cases.

Example:
- Correctly identifying healthy patients.

---

### 2. Bias and Variance

Bias and Variance are important concepts that affect model performance.

#### Bias

Bias occurs when a model is too simple and fails to capture underlying patterns.

Results in:
- Underfitting
- Poor training performance

---

#### Variance

Variance occurs when a model learns the training data too well.

Results in:
- Overfitting
- Poor performance on unseen data

---

### 3. Entropy

Entropy measures the uncertainty or randomness in a dataset.

Entropy Formula:

Entropy(S) = -Σ p(x) log₂ p(x)

Properties:
- High Entropy → High uncertainty
- Low Entropy → Low uncertainty

Used in:
- Decision Trees
- Information Gain

---

### 4. Mutual Information

Mutual Information measures how much information one variable provides about another.

Higher Mutual Information:
- Strong relationship

Lower Mutual Information:
- Weak relationship

Applications:
- Feature Selection
- Feature Importance Analysis

---

## Key Takeaways

✅ Sensitivity measures the ability to detect positive cases.

✅ Specificity measures the ability to detect negative cases.

✅ High Bias leads to underfitting.

✅ High Variance leads to overfitting.

✅ Entropy measures uncertainty.

✅ Mutual Information helps identify important features.

---

## Outcome

Learned evaluation metrics for classification problems, model generalization concepts, and foundational information theory concepts used in machine learning.