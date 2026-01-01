# Imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create simple linear data
m = 300
x = np.random.rand(m)
y = 4 * x + 3 + np.random.randn(m) * 0.5

#----------------------------------------------------

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2

    return cost / (2 * m)

#----------------------------------------------------

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

#----------------------------------------------------

def gradient_descent(x, y, w, b, alpha, num_iters):
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(x, y, w, b)
        J_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"Iter {i:4}: Cost {cost:.4f}, w {w:.4f}, b {b:.4f}")

    return w, b, J_history

#----------------------------------------------------

w_init = 0
b_init = 0
alpha = 0.1
iterations = 1000

w_final, b_final, J_history = gradient_descent(
    x, y, w_init, b_init, alpha, iterations
)

print(f"\nFinal parameters: w = {w_final:.3f}, b = {b_final:.3f}")

plt.scatter(x, y, label="Data")
plt.plot(x, w_final * x + b_final, color="red", label="Prediction")
plt.legend()
plt.show()
