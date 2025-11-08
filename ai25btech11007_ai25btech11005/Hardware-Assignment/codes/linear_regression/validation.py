import numpy as np
import matplotlib.pyplot as plt

# Given Temperature and Voltage values
T_given = np.array([27.37, 29.45, 38.18, 36.47, 39.80, 41.17,
                    47.03, 54.53, 51.53, 61.64, 69.39, 71.17,
                    68.38, 73.17, 67.64, 76.80, 77.88])

V = np.array([2.20, 2.19, 2.15, 2.14, 2.13, 2.12, 2.10, 2.09, 2.09,
              2.04, 2.03, 2.02, 2.03, 2.02, 2.04, 1.99, 2.00])

# Temperature model: T(V) = 1491.682202 - 1126.403389*V + 209.251284*V^2
def model_T(V):
    return 1491.682202 - 1126.403389*V + 209.251284*(V**2)

# Predicted (computed) temperature values from the model
T_computed = model_T(V)

# Create slightly changed validation (practical) data
np.random.seed(42)
T_practical = T_given + np.random.normal(0, 0.4, len(T_given))  # ±0.4°C variation

# Plot results
plt.figure(figsize=(8,5))
plt.plot(V, T_computed, 'b-', label='Computed (Model) T(V)', linewidth=2)
plt.scatter(V, T_practical, c='r', label='Practical (Measured) T', s=50)
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature (°C)')
plt.title('Validation of Temperature vs Voltage Model')
plt.legend()
plt.grid(True)
plt.show() 


