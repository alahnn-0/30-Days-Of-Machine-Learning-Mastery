# Day 8 of 30 — Gradient Descent + Linear Algebra Deep Dive

## 🎯 Focus Today
Continued math foundations — building on Day 7's intro to vectors/matrices with the operations that directly power optimization algorithms like gradient descent.

## 📌 Topics Covered

### 1. Gradient Descent
- The core idea: iteratively adjust parameters in the direction that *reduces* a cost function, using the negative gradient as the "downhill" direction.
- Update rule: `θ = θ - learning_rate * gradient`
- Why the learning rate matters — too small and convergence is painfully slow, too large and it can overshoot or diverge entirely.
- This is the algorithm you'll see reused everywhere from linear regression to training neural networks — same core idea, different scale.

### 2. Scalar Normalization
- Rescaling feature values (typically to a 0-1 range or a standard scale) so that no single feature dominates a model just because of its raw magnitude.
- Two common approaches:
  - **Min-Max scaling** — squashes values into `[0, 1]` based on the feature's min/max.
  - **Standardization (Z-score)** — rescales to mean 0, std dev 1: `(x - mean) / std`.
- **Why this connects to gradient descent:** unscaled features distort the cost function's shape (some directions much steeper than others), which makes gradient descent zig-zag and converge slower. Normalizing features speeds up and stabilizes convergence — this is the practical link between today's two big topics.

### 3. Linear Algebra — Vectors
- Reviewed vectors as ordered lists of numbers / points in space.
- **Vector scaling:** multiplying a vector by a scalar stretches or shrinks it without changing its direction (unless the scalar is negative, which flips it).
- **Vector multiplication:**
  - Dot product — combines two vectors into a single scalar, measuring alignment/similarity. This is the operation underneath every weighted sum in ML (e.g., `prediction = weights · features`).
  - Element-wise multiplication — multiplies corresponding entries, used constantly in vectorized NumPy operations instead of writing loops.

## 💡 Key Takeaway
Today's topics aren't separate — they connect directly: **normalization reshapes the cost surface**, and **gradient descent** is the algorithm that walks down that surface using **vector operations** (scaling, dot products) at every single step. Understanding gradient descent without the vector math underneath it is just memorizing a formula; today filled in the "why" behind the "how."

## ⏭️ Next Up
- Matrix multiplication and how it generalizes the dot product across many samples at once (this is how gradient descent gets vectorized for an entire dataset instead of looping row by row)
- Apply normalized gradient descent to finish the Linear Regression fit on the Housing Prices dataset

---
*Day 8 of 30 — ML Challenge*
