import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load first dataset (data1.csv) ---
data = np.loadtxt("/home/dhanush-kumar-a/EE1030-2025-Project-Submissions/ai25btech11010_ai25btech11014/Hardware-Assignment/codes/linear_regression/data1.csv", delimiter=',', skiprows=1)
T = data[:, 0]
V = data[:, 1]

# Fit model for data1.csv
X = np.vstack((np.ones_like(T), T, T**2)).T
a0, a1, a2 = np.linalg.lstsq(X, V, rcond=None)[0]

print("=== Least Squares Fit Results (data1.csv) ===")
print(f"a0 = {a0:.6f}")
print(f"a1 = {a1:.6e}")
print(f"a2 = {a2:.6e}")
print(f"Model: V = {a0:.6f} + {a1:.6e}*T + {a2:.6e}*T^2\n")

# Generate fitted curve
T_fit = np.linspace(min(T), max(T), 200)
V_fit = a0 + a1 * T_fit + a2 * (T_fit ** 2)

# --- Load second dataset (data2.csv) ---
data2 = np.loadtxt("/home/dhanush-kumar-a/EE1030-2025-Project-Submissions/ai25btech11010_ai25btech11014/Hardware-Assignment/codes/linear_regression/data2.csv", delimiter=',', skiprows=1)
T2 = data2[:, 0]
V2 = data2[:, 1]

# --- Plot ---
plt.figure(figsize=(8, 5))

# Plot only the fitted curve from data1.csv
plt.plot(T_fit, V_fit, color='blue', linewidth=2, label='Fitted Curve (data1.csv)')

# Plot only the points from data2.csv
plt.scatter(T2, V2, color='red', label='Data Points (data2.csv)', marker='o')

plt.xlabel("Temperature (T)")
plt.ylabel("Voltage (V)")
plt.title("Temperature vs Voltage")
plt.legend()
plt.grid(True)

# Save figure
save_path = "/home/dhanush-kumar-a/EE1030-2025-Project-Submissions/ai25btech11010_ai25btech11014/Hardware-Assignment/figs/T_vs_V_fit_with_data2_points.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

