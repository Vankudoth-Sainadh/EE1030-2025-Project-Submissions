import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual readings)
actual_temp = np.array([38.4, 39.1, 40.2, 41.0, 42.3, 43.2, 44.5])
calculated_temp = np.array([38.5, 39.3, 40.4, 41.2, 42.1, 43.3, 44.2])

# Create plot
plt.figure(figsize=(8,5))
plt.scatter(actual_temp, calculated_temp, color='orange', label='Data Points')

# Plot ideal y=x line
plt.plot(actual_temp, actual_temp, color='orange', linestyle='--', label='Ideal (y=x)')

# Labels and title
plt.title('Calculated vs Actual Temperature', fontsize=14)
plt.xlabel('Actual Temperature (°C)', fontsize=12)
plt.ylabel('Calculated Temperature (°C)', fontsize=12)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Display the plot
plt.show()
