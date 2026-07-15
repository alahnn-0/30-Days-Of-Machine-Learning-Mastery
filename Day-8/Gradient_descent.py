import numpy as np

# 1. Create a simple toy dataset (similar to our widget)
# x = house size (in 1000s of sq ft), y = price (in $100,000s)
X_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([1.5, 2.5, 3.2, 4.5, 5.0])

# 2. Define the Cost Function J(w,b)
def compute_cost(X, y, w, b):
    m = len(X)
    predictions = w * X + b
    # Squared differences
    squared_errors = (predictions - y) ** 2
    # J(w,b) = 1/(2m) * sum(squared_errors)
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    return cost

# 3. Calculate the gradients (partial derivatives)
def compute_gradient(X, y, w, b):
    m = len(X)
    predictions = w * X + b
    
    # Derivates: dJ/dw and dJ/db
    dj_dw = (1 / m) * np.sum((predictions - y) * X)
    dj_db = (1 / m) * np.sum(predictions - y)
    
    return dj_dw, dj_db

# 4. Run Gradient Descent
def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    w = w_init
    b = b_init
    
    # Keep track of the cost history to verify it goes down
    cost_history = []
    
    for i in range(num_iters):
        # Calculate gradients
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
        # Update parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost for tracking
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        
        # Print progress every 10% of iterations
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4d}: Cost = {cost:8.4f} | w = {w:6.3f}, b = {b:6.3f}")
            
    return w, b, cost_history

# --- Let's Run It! ---
# Starting with bad initial guesses
w_start = 0.0
b_start = 0.0
learning_rate = 0.05
iterations = 500

print("Starting Gradient Descent...")
w_final, b_final, history = gradient_descent(X_train, y_train, w_start, b_start, learning_rate, iterations)

print("\n--- Training Complete ---")
print(f"Learned Weight (w): {w_final:.3f}")
print(f"Learned Bias (b):   {b_final:.3f}")
print(f"Final Cost:         {history[-1]:.4f}")