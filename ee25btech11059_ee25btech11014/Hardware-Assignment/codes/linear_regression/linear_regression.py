import numpy as np

# Given data
T = np.array([34.3, 29.8, 48.6, 81.7, 38.55, 53.3, 61.9, 93.6, 89.2])  # °C
V = np.array([1.764, 1.720, 1.800, 1.890, 1.769, 1.813, 1.857, 1.960, 1.930])  # V

# Design matrix for quadratic fit
X = np.vstack([np.ones_like(V), V, V**2]).T

# Solve least squares
a, b, c = np.linalg.lstsq(X, T, rcond=None)[0]

# Print the fitted equation
print(f"T(V) = {a:.6f} + ({b:.6f})V + ({c:.6f})V^2")