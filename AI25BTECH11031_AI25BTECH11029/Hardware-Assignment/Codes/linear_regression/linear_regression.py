import numpy as np

T = np.array([26.5, 33.6, 37.5, 41.2, 47.6, 53.5, 57.2, 62.5, 67.4, 72.1, 
              77.6, 82.5, 89.5, 94.8, 30.5, 36.4, 40.2, 43.4])
V = np.array([2.224, 2.204, 2.19, 2.18, 2.155, 2.126, 2.092, 2.067, 2.058, 2.038, 
              2.023, 1.994, 1.979, 1.955, 2.214, 2.224, 2.185, 2.18])

X_T = np.vstack([V**2, V, np.ones_like(V)]).T
coeffs_T = np.linalg.inv(X_T.T @ X_T) @ X_T.T @ T
a, b, c = coeffs_T

X_V = np.vstack([T**2, T, np.ones_like(T)]).T
coeffs_V = np.linalg.inv(X_V.T @ X_V) @ X_V.T @ V
p, q, r = coeffs_V

print("Equation 1: Temperature as a function of Voltage")
print(f"T = {a:.4f} * V² + ({b:.4f}) * V + ({c:.4f})\n")

print("Equation 2: Voltage as a function of Temperature")
print(f"V = {p:.6f} * T² + ({q:.6f}) * T + ({r:.6f})")
