import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Experimental data
# -----------------------------
T = np.array([97.9, 97.8, 98.5, 91.6, 88.1, 80.1, 73.9, 71.3, 68.4, 66.3,
              63.2, 60.3, 57.5, 55.0, 53.4, 53.1, 52.1, 49.6, 42.6, 40.8,
              40.3, 40.0, 39.1])   # Temperature (°C)
V = np.array([1.96, 1.95, 1.94, 1.93, 1.92, 1.90, 1.89, 1.87, 1.86, 1.85,
              1.84, 1.83, 1.82, 1.81, 1.80, 1.78, 1.79, 1.76, 1.74, 1.75,
              1.72, 1.70, 1.69])   # Voltage (V)

# -----------------------------
# Fit a 2nd-degree polynomial:  V = b2*T^2 + b1*T + b0
# -----------------------------
coeffs = np.polyfit(T, V, 2)
b2, b1, b0 = coeffs
print(f"Equation: V = {b2:.5e}T² + {b1:.5e}T + {b0:.5e}")

# -----------------------------
# Generate smooth fitted curve
# -----------------------------
T_fit = np.linspace(min(T), max(T), 200)
V_fit = np.polyval(coeffs, T_fit)

# -----------------------------
# Plot measured data and fitted curve
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(T, V, color='red', label='Measured Data')
plt.plot(T_fit, V_fit, color='black', label='Fitted Curve')
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Output Voltage (V)', fontsize=12)
plt.title('Temperature vs Output Voltage (PT100 Sensor)', fontsize=13)
plt.legend()
plt.grid(True)

# Add fitted equation on plot
eq_text = f"V = {b2:.5e}T² + {b1:.5e}T + {b0:.5e}"
plt.text(45, 1.93, eq_text, color='blue', fontsize=10)

# -----------------------------
# Save figure
# -----------------------------
plt.savefig("figs/PT100_calibration_curve.png", dpi=300, bbox_inches='tight')
plt.show()
