import numpy as np
import matplotlib.pyplot as plt

# Load training data

A = np.loadtxt('../data/data.txt')
T_train = A[:, 0]  # Temperature
V_train = A[:, 1]  # Voltage

#V = a + bT + cT²

X_train = np.column_stack((np.ones_like(T_train), T_train, T_train**2))
theta_VT = np.linalg.lstsq(X_train, V_train, rcond=None)[0]
print("Coefficients for V(T):", theta_VT)

# Plot V vs T
plt.figure()
plt.plot(T_train, V_train, 'ko', label='Training Data')
plt.plot(T_train, X_train @ theta_VT, 'r-', label='Predicted Model (V vs T)')
plt.grid()
plt.xlabel('Temperature(C)')
plt.ylabel('Voltage(V)')
plt.legend()
plt.savefig('../figs/train_V_vs_T.png')
plt.close()

#T = a + bV + cV²
Y_train = np.column_stack((np.ones_like(V_train), V_train, V_train**2))
theta_TV = np.linalg.lstsq(Y_train, T_train, rcond=None)[0]
print("Coefficients for T(V):", theta_TV)

# Plot T vs V
plt.figure()
plt.plot(V_train, T_train, 'ko', label='Training Data')
plt.plot(V_train, Y_train @ theta_TV, 'b-', label='Predicted Model (T vs V)')
plt.grid()
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.savefig('../figs/train_T_vs_V.png')
plt.close()

# Validation data
B = np.loadtxt('../data/validation_data.txt')
V_valid = B[:, 1]
T_valid = B[:, 0]

Y_valid = np.column_stack((np.ones_like(V_valid), V_valid, V_valid**2))

# Plot validation
plt.figure()
plt.plot(V_valid, T_valid, 'ko', label='Validation Data')
plt.plot(V_valid, Y_valid @ theta_TV, 'r-', label='Predicted T(V)')
plt.grid()
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.savefig('../figs/validation.png')
plt.close()

