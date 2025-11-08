import numpy as np
import matplotlib.pyplot as plt
import os

# Load data from CSV file
data = np.loadtxt("/home/dhanush-kumar-a/EE1030-2025-Project-Submissions/ai25btech11010_ai25btech11014/Hardware-Assignment/codes/linear_regression/data1.csv", delimiter=',', skiprows=1)
T = data[:, 0]
V = data[:, 1]

# Model: V = a0 + a1*T + a2*T^2
X = np.vstack((np.ones_like(T), T, T**2)).T
a0, a1, a2 = np.linalg.lstsq(X, V, rcond=None)[0]

print("Least Squares Fit Results:")
print(f"a0 = {a0:.6f}")
print(f"a1 = {a1:.6e}")
print(f"a2 = {a2:.6e}")
print(f"\nModel: V(T) = {a0:.6f} + {a1:.6e}*T + {a2:.6e}*T^2")

# Generate fitted curve
T_fit = np.linspace(min(T), max(T), 200)
V_fit = a0 + a1 * T_fit + a2 * (T_fit ** 2)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(T, V, color='red', label='Data Points', marker='o')
plt.plot(T_fit, V_fit, color='blue', linewidth=2, label='Fitted Curve')

plt.xlabel("Temperature (T)")
plt.ylabel("Voltage (V)")
plt.title("Temperature vs Voltage (Least Squares Fit)")
plt.legend()
plt.grid(True)

# Ensure save directory exists
save_dir = "/home/dhanush-kumar-a/EE1030-2025-Project-Submissions/ai25btech11010_ai25btech11014/Hardware-Assignment/figs"

# Save figure
save_path = os.path.join(save_dir, "T_vs_V_fit.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

