import matplotlib.pyplot as plt
import numpy as np

# Data points
temperature = np.array([61.2, 59.6, 58.6, 50.6, 49.6, 75.8, 78.8, 80.2, 
                        84.6, 83.9, 82.5, 55.6, 46.9, 57.9, 51.2, 
                        48.7, 84.4, 83.5, 80.9])
voltage = np.array([1.7619, 1.7564, 1.7551, 1.7473, 1.7458, 1.7885, 
                    1.7969, 1.8003, 1.8137, 1.8112, 1.8073, 1.7512, 
                    1.7427, 1.7541, 1.7483, 1.7445, 1.8131, 1.8104, 1.8021])

# Polynomial fit (4th order for smooth curvature)
coeffs = np.polyfit(temperature, voltage, 4)
poly_fit = np.poly1d(coeffs)
t_fit = np.linspace(min(temperature), max(temperature), 300)
v_fit = poly_fit(t_fit)

# Plot Temperature vs Voltage
plt.figure(figsize=(8, 6))
plt.scatter(temperature, voltage, color='black', s=50, label='Measured Data', marker='x')
plt.plot(t_fit, v_fit, color='deepskyblue', linewidth=3, label='Best-Fitted Curve')
plt.xlabel("Temperature (°C)", fontsize=13)
plt.ylabel("Output Voltage (V)", fontsize=13)
plt.title("Temperature vs Output Voltage", fontsize=15, weight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()