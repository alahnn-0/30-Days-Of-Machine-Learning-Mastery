# Day 7 — End-to-End ML Project: Housing Prices (Part 1)

**Reference:** *Hands-On Machine Learning with Scikit-Learn & TensorFlow* — Aurélien Géron, Ch. 2

## 🎯 What I Worked On
Started an end-to-end ML project predicting California housing prices, following Géron's structured project workflow. Focus today: data acquisition, exploratory data analysis (EDA), and — the deep dive — **train/test splitting strategies**.

## 📌 Topics Covered

### 1. Data Acquisition
- Automated dataset download + extraction (`tarfile`, `urllib.request`)
- Cached locally so it doesn't re-download on every run

### 2. Exploratory Data Analysis (EDA)
- `df.info()` → caught a missing-data issue: `total_bedrooms` has fewer non-null values than the rest
- `df.describe()` → checked scale/range differences across features
- `value_counts()` on `ocean_proximity` → explored the one categorical column
- `df.hist(bins=50)` → visualized distributions for every numeric feature at once, spotted skew and capped values

### 3. Train/Test Splitting — Three Approaches
This was the core learning today — understanding *why* naive splitting fails before reaching for the library function:

| Approach | Method | Limitation |
|---|---|---|
| Pure random shuffle | `np.random.permutation` | No fixed seed → different split every run, not reproducible |
| Hash-based split | `crc32` hash on a unique row ID | Stable across re-runs / new data, but needs a good unique ID |
| `sklearn.train_test_split` | Built-in, `random_state=42` | Production-ready, reproducible — what I'd actually use |

**Key insight:** hash-based splitting keeps previously-assigned rows in the same split even if new data is added later — pure random shuffling would reshuffle everything and risk leaking old test rows into training.

### 4. Stratified Sampling (setup)
- Binned continuous `median_income` into 5 categories (`income_cat`) using `pd.cut`
- **Why:** income is a strong predictor of price — a random split could accidentally skew the test set's income distribution and bias evaluation
- Sets up `StratifiedShuffleSplit` (next step) to preserve income-category proportions in both train and test sets

## 💡 Key Takeaways
- EDA before anything else — `info()` and `describe()` catch missing data and scale issues that affect every downstream step
- Not all train/test splits are equal — reproducibility and stability across dataset updates matter, not just "get a train and test set"
- Stratified sampling exists to fix a real bias risk, not just as textbook ceremony

## ⏭️ Next Up
- Apply `StratifiedShuffleSplit` on `income_cat`
- Correlation analysis + scatter matrix
- Handle missing `total_bedrooms` values (imputation)
- Encode `ocean_proximity` (categorical → numeric)
- Feature scaling

---
*Part of my 52-day ML roadmap — Beginner to Advanced*
