import numpy as np
import matplotlib.pyplot as plt

validation_voltage = np.array([2.66, 2.67, 2.68, 2.69, 2.71, 2.74, 2.75, 2.77, 2.79, 2.81, 2.82, 2.83, 2.85, 2.87, 2.89])
validation_temperature = np.array([32, 36.5, 41, 45.5, 50.5, 56.6, 62, 67.5,71.5, 76.6, 81.12, 85.9, 90.7, 95.5, 97.5])


def temperature_from_voltage(V):
    return (65.976142 +
            10.543790 * ((V - 2.768667) / 0.073200) +
            10.482903 * ((V * V - 7.670873) / 0.405495))


voltage_range = np.linspace(min(validation_voltage), max(validation_voltage), 200)
temp_curve = temperature_from_voltage(voltage_range)

#plt.style.use('dark_background')
plt.figure(figsize=(8,6))

plt.plot(validation_temperature, validation_voltage, 'o', color='red', markersize=7, label='Validation Data')

plt.plot(temp_curve, voltage_range, color='deepskyblue', label='Model Equation Fit')

plt.xlabel("Temperature (°C)")
plt.ylabel("Voltage (V)")
plt.title("Validation Data and Model Equation Fit")
plt.legend()
plt.grid(True)
plt.show()

