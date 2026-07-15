import numpy as np

# A raw feature matrix: [Size (sq ft), Bedrooms]
X_raw = np.array([
    [600.0,  1.0],
    [1200.0, 2.0],
    [1800.0, 3.0],
    [2400.0, 3.0],
    [3000.0, 4.0]
])

def zscore_normalize(X):
    # Calculate mean along columns (axis=0)
    mu = np.mean(X, axis=0)
    
    # Calculate standard deviation along columns (axis=0)
    sigma = np.std(X, axis=0)
    
    # Apply standardizing formula
    X_scaled = (X - mu) / sigma
    
    return X_scaled, mu, sigma

# Scale the data
X_norm, mu, sigma = zscore_normalize(X_raw)

print("Original Data:\n", X_raw)
print("\n--- Normalized Data ---")
print("Mean of each feature:   ", mu)
print("Std Dev of each feature:", sigma)
print("\nScaled Matrix:\n", np.round(X_norm, 3))