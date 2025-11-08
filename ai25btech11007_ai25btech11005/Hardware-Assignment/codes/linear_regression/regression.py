import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# --- Data from the original prompt ---
# Voltage (V) and Temperature (T) data points
V_data = np.array([2.20, 2.19, 2.15, 2.14, 2.13, 2.12, 2.10, 2.09, 2.09,
                   2.04, 2.03, 2.02, 2.03, 2.02, 2.04, 1.99, 2.00])
T_data = np.array([27.5, 26.5, 34.8, 37.9, 41.8, 45.8, 50.7, 54.9, 58,
                   60.3, 63.1, 61.6, 63.8, 66, 69.6, 85, 81.3])

# --- Regression for T = c0 + c1*V + c2*V^2 ---

# 1. Design Matrix (B)
# Independent variable is V. Columns are [1, V, V^2]
B = np.column_stack((np.ones_like(V_data), V_data, V_data**2))

# 2. Least Squares Solution
# Solve the system B * X = T_data for X = [c0, c1, c2]
# Using np.linalg.lstsq for stable calculation
X, residuals, rank, singular_values = la.lstsq(B, T_data, rcond=None)
c0, c1, c2 = X.flatten()

print("--- Regression Results ---")
print(f"Coefficients (c0, c1, c2): {X.flatten()}")
print(f"Temperature Model: T(V) = {c0:.6f} + {c1:.6f}*V + {c2:.6f}*V^2")

# --- Plotting ---

# Generate fitted curve
v_fit = np.linspace(np.min(V_data), np.max(V_data), 500)
t_fit = c0 + c1 * v_fit + c2 * v_fit**2

plt.figure(figsize=(10, 6))
# Plot the fitted curve (V on x-axis, T on y-axis)
plt.plot(v_fit, t_fit, label='Fitted Curve: $T(V)$', color='blue', linewidth=2)
# Plot the data points
plt.scatter(V_data, T_data, c='r', marker='o', label='Data points')

plt.xlabel('Voltage (V)')
plt.ylabel('Temperature (T)')
plt.title('Quadratic Regression: Temperature vs. Voltage')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
