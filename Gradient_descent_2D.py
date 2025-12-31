# Imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

m = 300
X = np.random.rand(m, 2)     # two features
y = 3*X[:,0] + 5*X[:,1] + 2 + np.random.randn(m)*0.5
#----------------------------------------------------

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w[0]*X[i,0] + w[1]*X[i,1] + b
        cost += (f_wb - y[i])**2

    return cost / (2*m)

#----------------------------------------------------

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    dj_dw = np.zeros(2)
    dj_db = 0

    for i in range(m):
        f_wb = w[0]*X[i,0] + w[1]*X[i,1] + b
        error = f_wb - y[i]

        dj_dw[0] += error * X[i,0]
        dj_dw[1] += error * X[i,1]
        dj_db += error

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

#----------------------------------------------------

def gradient_descent(X, y, w, b, alpha, num_iters):
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"Iter {i:4}: Cost {cost:.4f}, w {w}, b {b:.4f}")

    return w, b, J_history

#----------------------------------------------------

w_init = np.zeros(2)
b_init = 0
alpha = 0.1
iterations = 1000

w_final, b_final, J_hist = gradient_descent(
    X, y, w_init, b_init, alpha, iterations
)

print(f"\nFinal parameters:")
print(f"w = {w_final}")
print(f"b = {b_final}")

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y, label="Data")

x1, x2 = np.meshgrid(
    np.linspace(0,1,20),
    np.linspace(0,1,20)
)
y_pred = w_final[0]*x1 + w_final[1]*x2 + b_final

ax.plot_surface(x1, x2, y_pred, alpha=0.5)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

plt.show()
