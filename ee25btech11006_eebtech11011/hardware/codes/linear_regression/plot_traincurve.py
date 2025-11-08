import numpy as np
import matplotlib.pyplot as plt

temp = np.array([48.3, 52.1, 56.3, 61.5, 65.5, 74.6, 79.2, 84.3, 90.2, 95.2])
volt = np.array([1.183, 1.193, 1.202, 1.237, 1.256, 1.276, 1.285, 1.295, 1.300, 1.315])

coeffs = np.polyfit(temp, volt, 2)
poly = np.poly1d(coeffs)

temp_smooth = np.linspace(min(temp), max(temp), 200)
volt_smooth = poly(temp_smooth)

plt.scatter(temp, volt, color='blue', label='Measured Data', zorder=5)
plt.plot(temp_smooth, volt_smooth, color='red', label='Curve', linewidth=2)
plt.xlabel("Temperature (°C)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Temperature")
plt.legend()
plt.grid(True)
plt.show()

print("Quadratic Fit Equation:")
print(f"Voltage = {coeffs[0]:.8f}*T² + {coeffs[1]:.8f}*T + {coeffs[2]:.8f}")