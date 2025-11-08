import numpy as np
import matplotlib.pyplot as plt

# Combine all data
T = np.array([
    68.9, 59, 57.8, 57, 56.6, 55.5, 55.2, 54.5,
    53.9, 53.5, 50.2, 48.7, 48.3, 47.6, 44.8, 44.8, 44.1,
    44.4, 43.3, 42.3, 41.2, 40.3, 39.1, 38.3
])
V = np.array([
    1.872, 1.867, 1.862, 1.857, 1.852, 1.848, 1.843, 1.838,
    1.833, 1.828, 1.823, 1.818, 1.813, 1.808, 1.804, 1.799, 1.794,
    1.794, 1.789, 1.784, 1.779, 1.774, 1.769, 1.764
])

# Fit quadratic model V = a + bT + cT^2
X = np.vstack([np.ones(len(T)), T, T**2]).T
coeffs = np.linalg.inv(X.T @ X) @ X.T @ V
a, b, c = coeffs

# Generate smooth curve
T_fit = np.linspace(min(T), max(T), 300)
V_fit = a + b*T_fit + c*(T_fit**2)

# Plot (Temperature on X-axis, Voltage on Y-axis)
plt.figure(figsize=(8,6))
plt.scatter(T, V, color='red', s=40, label='Measured Data')
plt.plot(T_fit, V_fit, color='blue', linewidth=2, label='Best Fit Curve')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Output Voltage (V)', fontsize=12)
plt.title('Output Voltage vs Temperature', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")
print(f"\nEquation: V(T) = {a:.4f} + {b:.5f}T + {c:.6f}T²")