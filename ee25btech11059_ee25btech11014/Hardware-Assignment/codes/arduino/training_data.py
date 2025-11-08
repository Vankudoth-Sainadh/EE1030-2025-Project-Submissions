import numpy as np
import matplotlib.pyplot as plt

# Given data
T = np.array([34.30, 29.80, 48.60, 81.70, 38.55, 53.30, 61.90, 93.60, 89.20])  # °C
V = np.array([1.764, 1.720, 1.800, 1.890, 1.769, 1.813, 1.857, 1.960, 1.930])  # Volts

# Quadratic least squares fit: V = a + bT + cT^2
A = np.vstack([np.ones(len(T)), T, T**2]).T
coeff, _, _, _ = np.linalg.lstsq(A, V, rcond=None)
a, b, c = coeff

# Generate smooth curve for plotting
T_fit = np.linspace(min(T), max(T), 200)
V_fit = a + b*T_fit + c*(T_fit**2)

# Plot
plt.figure(figsize=(8,5))
plt.scatter(T, V, color='red', label='Measured Data')
plt.plot(T_fit, V_fit, color='blue', label='Least Squares Fit')
plt.title("Output Voltage vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Output Voltage (V)")
plt.grid(True)
plt.legend()
plt.show()