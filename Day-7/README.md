# Day 7 of 30 — Linear Algebra + Housing Prices Project

**Two tracks today:** Math foundations (Linear Algebra) + the ongoing Housing Prices ML project.

---

## Part A: Math — Linear Algebra (Vectors & Matrices)

**Vectors**
- An ordered list of numbers representing a point or direction in space — e.g., a row of features in a dataset is literally a vector.
- Key operations: addition, scalar multiplication, dot product (measures how much two vectors "align" — the basis for a lot of similarity/distance calculations in ML).
- Norms (length of a vector): L1 (sum of absolute values) and L2 (Euclidean length) — these show up constantly as regularization terms (Lasso/Ridge) and distance metrics (KNN).

**Matrices**
- A 2D grid of numbers — in ML, your entire dataset (`housing` DataFrame, minus the target) is essentially a matrix: rows = samples, columns = features.
- Key operations: addition, scalar multiplication, matrix multiplication (combines/transforms data — this is literally what a neural network layer does), transpose (flip rows/columns), inverse (used in the closed-form solution to linear regression, the Normal Equation).
- Matrix multiplication is *not* commutative (`A @ B ≠ B @ A`) — a common early trip-up, worth internalizing early since it affects how you order operations later (e.g., in backprop).

**Why this connects directly to today's project:** every column operation you did on the `housing` DataFrame (ratios, scaling, etc.) is vector/matrix math under the hood — pandas and NumPy are just giving you a friendly interface over linear algebra operations.

---

## Part B: End-to-End ML Project: California Housing Prices

**Reference:** *Hands-On Machine Learning with Scikit-Learn & TensorFlow* — Aurélien Géron, Ch. 2
**Notebook goal:** Predict median house value from housing/location features, following a structured end-to-end ML workflow.

---

## 1. Setup & Data Acquisition
```python
from pathlib import Path
import pandas as pd, tarfile, urllib.request
import matplotlib.pyplot as plt
import numpy as np
```
`load_housing_data()` downloads a `.tgz` archive from GitHub, extracts it, and reads the CSV — but only if it isn't already on disk (`tarball_path.is_file()` check). This is a **caching pattern**: avoids re-downloading every time the notebook runs, which matters once your dataset gets large or your connection is slow.

## 2. Exploratory Data Analysis (EDA)

**`housing.info()`** — Reveals two issues immediately:
- `total_bedrooms` has only 20,433 non-null entries (fewer than total rows) → **missing data** to handle later.
- `ocean_proximity` is dtype `object` → a **categorical** feature that will need encoding before any model can use it (models only understand numbers).

**`value_counts()` on `ocean_proximity`** — Counts how many districts fall into each category (`<1H OCEAN`, `INLAND`, `NEAR BAY`, etc.). Useful before choosing an encoding strategy — if one category is extremely rare, that affects how you split/encode it.

**`housing.describe()`** — Summary stats (mean, std, min, max, quartiles) for every numeric column. This is where you first notice **scale mismatches** between features (e.g., `median_income` ranges ~0-15 while `total_rooms` can be in the thousands) — a preview of why feature scaling matters later.

**`housing.hist(bins=50, figsize=(20,15))`** — Plots every numeric column's distribution at once. Purpose: spot skew, capped values (house values/ages often hit a hard ceiling in this dataset — a data quality issue, not a real-world pattern), and scale differences visually.

---

## 3. Train/Test Splitting — Three Approaches, Built Up in Order

**Approach 1 — Pure random shuffle (`shuffle_and_split_data`)**
Shuffles row indices with `np.random.permutation`, slices off a test fraction. **Problem:** no fixed seed, so every run gives a different split — bad for reproducibility, and every re-run risks the model eventually "seeing" the entire dataset across experiments (data leakage over time).

**Approach 2 — Hash-based split (`is_id_in_test_set` + `split_data_with_id_hash`)**
Uses `crc32` to hash a row's unique ID; the hash value deterministically decides train vs. test. **Why this is smarter:** if you add new rows later, old rows keep their original split assignment — random shuffling would reshuffle everything and could leak former test rows into training.
- First attempt used `housing.reset_index()` as the ID source — fragile, since row order/index isn't guaranteed stable if data is ever re-fetched or rows are added/removed upstream.
- Fix: build a real composite key — `longitude * 1000 + latitude` — tying the ID to something inherently stable about each district (its location).

**Approach 3 — `sklearn.train_test_split`**
Same random-split idea as Approach 1, but with `random_state=42` for reproducibility. This is what's actually used going forward — the earlier two were there to build understanding of *why* naive splitting is risky before reaching for the library shortcut.

---

## 4. Stratified Sampling — Fixing a Real Bias Risk

**Binning income:**
```python
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
```
`median_income` is a strong predictor of house price. A purely random split *could* accidentally produce a test set with a skewed income distribution, silently biasing your evaluation. Binning it into 5 categories lets you control for this.

**Two equivalent ways to stratify**, both included in the notebook:
- `StratifiedShuffleSplit` (explicit, verbose — loops over the one split it generates)
- `train_test_split(..., stratify=housing["income_cat"])` (same result, one line — this is the one you'd actually use)

**Verification:** `strat_test_set["income_cat"].value_counts() / len(strat_test_set)` — checks that the test set's income-category proportions match the full dataset. This is the actual payoff of stratification — confirming it worked, not just trusting it did.

**Cleanup:** `income_cat` is dropped from both sets afterward — it was scaffolding to build the split, not a real feature; leaving it in would be redundant with `median_income` itself.

---

## 5. Geographic Visualization

Three progressively richer scatter plots of `longitude` vs `latitude`:
1. Plain scatter — shows California's shape, nothing more.
2. `alpha=0.2` — transparency reveals **density** (overlapping points darken), showing where districts cluster (Bay Area, LA, etc.) that a solid-dot plot hides.
3. Full version — point **size** encodes `population`, point **color** encodes `median_house_value` (`cmap="jet"`). This single plot visually confirms two things: coastal areas are more expensive, and high-population districts cluster in specific corridors — motivating `ocean_proximity` and location as important features before you've run a single model.

---

## 6. Correlation Analysis

**`housing.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)`** — Ranks every numeric feature by linear correlation with the target. `median_income` comes out as the strongest predictor — matches the visual pattern from the scatter plot above.

**`scatter_matrix`** on `["median_house_value", "median_income", "total_rooms", "housing_median_age"]` — Plots every pairwise combination of these features against each other. Purpose: correlation coefficients only capture *linear* relationships — a scatter matrix lets you visually catch non-linear patterns or weird artifacts (like the capped price ceiling showing up as a hard horizontal line).

**Focused scatter:** `median_income` vs `median_house_value` alone — confirms the strong, mostly-linear relationship, but also shows the capped-value artifact clearly (a flat horizontal band at the top).

---

## 7. Feature Engineering

Three new features built from ratios of existing columns:
```python
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
```
**Why this matters:** raw totals (`total_rooms`, `total_bedrooms`) are only meaningful relative to how many households they're spread across — a district with 1000 rooms and 500 households is very different from one with 1000 rooms and 100 households. Turning totals into per-household ratios often produces features with *stronger* correlation to the target than the raw columns did — re-running `corr()` after this step confirms whether that happened here.

---

## 8. Preparing for Modeling

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```
Final split of the **training set** into `housing` (features/predictors, `X`) and `housing_labels` (target, `y`) — the standard shape scikit-learn expects for `.fit(X, y)`. This is the clean handoff point from EDA into actual preprocessing pipelines (imputation, encoding, scaling) and model training.

---

## Key Takeaways from Today
- EDA isn't optional ceremony — `info()` and `describe()` caught real issues (missing data, categorical column, scale mismatch) that shape every decision after.
- Not all train/test splits are equal — stability across dataset updates and avoiding sampling bias (stratification) both matter, not just "get *a* split."
- Visualization is doing analytical work here, not decoration — the geographic plot and scatter matrix each revealed patterns correlation coefficients alone would've hidden.
- Feature engineering (ratios over raw totals) can beat raw columns on correlation strength — always worth testing before feeding categories or the target and features are correctly separated. 

## Next Up
- Handle missing `total_bedrooms` (imputation)
- Encode `ocean_proximity` (one-hot)
- Feature scaling
- Build first preprocessing pipeline + baseline model (Linear Regression)

---
*Day 7 of 30 — ML Challenge*
