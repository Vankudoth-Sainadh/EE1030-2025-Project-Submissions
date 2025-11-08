import numpy as np
import matplotlib.pyplot as plt

temp = np.array([43.86, 44.3, 76.49, 83.29, 86.41, 89.09, 91.84])
volt = np.array([1.178, 1.183, 1.291, 1.301, 1.311, 1.316, 1.321])

sorted_idx = np.argsort(temp)
temp = temp[sorted_idx]
volt = volt[sorted_idx]

coeffs = np.polyfit(temp, volt, 2)
poly = np.poly1d(coeffs)

temp_smooth = np.linspace(40, 95, 300)
volt_smooth = poly(temp_smooth)

plt.figure(figsize=(8,6))
plt.plot(temp_smooth, volt_smooth, color='darkorange', linewidth=2, label='Curve Fit')
plt.scatter(temp, volt, color='purple', s=60, label='Measured Data')

plt.xlabel("Temperature (°C)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Temperature")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.xlim(40, 95)
plt.ylim(1.17, 1.33)
plt.xticks(np.arange(40, 100, 5))
plt.yticks(np.arange(1.17, 1.34, 0.01))

plt.show()

print("Quadratic Fit Equation:")
print(f"Voltage = {coeffs[0]:.8f}*T² + {coeffs[1]:.8f}*T + {coeffs[2]:.8f}")