import numpy as np
import time

# Let's say we have 1 million features (a massive dataset)
w = np.random.rand(1000000)
x = np.random.rand(1000000)
b = 0.5

# -----------------------------------------------------
# Approach 1: The un-vectorized way (Using a For Loop)
# -----------------------------------------------------
tic = time.time()

prediction = 0
for j in range(len(w)):
    prediction += w[j] * x[j]
prediction += b

toc = time.time()
print(f"For loop prediction: {prediction:.2f}")
print(f"Time taken: {1000 * (toc - tic):.2f} ms\n")


# -----------------------------------------------------
# Approach 2: The Vectorized way (Using NumPy Dot Product)
# -----------------------------------------------------
tic = time.time()

# This one line replaces the entire for loop!
prediction_vec = np.dot(w, x) + b 

toc = time.time()
print(f"Vectorized prediction: {prediction_vec:.2f}")
print(f"Time taken: {1000 * (toc - tic):.2f} ms")