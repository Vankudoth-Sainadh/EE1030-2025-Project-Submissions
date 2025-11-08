
# Polynomial Regression for PT100 Sensor Calibration

import numpy as np
import pandas as pd

# Step 1: Load calibration dataset
dataset = pd.read_csv("training2.csv")

# Extract temperature and corresponding voltage
temperature_vals = dataset["T"].to_numpy()
voltage_vals = dataset["V"].to_numpy()

# Step 2: Fit a second-order polynomial (Voltage as a function of Temperature)
# Model:  V = c0 + c1*T + c2*T^2
poly_terms = np.column_stack((np.ones(len(temperature_vals)), temperature_vals, temperature_vals**2))
voltage_coeffs, *_ = np.linalg.lstsq(poly_terms, voltage_vals, rcond=None)

print(f"Voltage model coefficients: {voltage_coeffs}")
print(f"V(T) = {voltage_coeffs[0]:.6f} + {voltage_coeffs[1]:.6e}*T + {voltage_coeffs[2]:.6e}*T^2")

# Step 3: Invert the model (Temperature as a function of Voltage)
# Model: T = b0 + b1*V + b2*V^2
inverse_terms = np.column_stack((np.ones(len(voltage_vals)), voltage_vals, voltage_vals**2))
temp_coeffs, *_ = np.linalg.lstsq(inverse_terms, temperature_vals, rcond=None)

print(f"Temperature model coefficients: {temp_coeffs}")
print(f"T(V) = {temp_coeffs[0]:.6f} + {temp_coeffs[1]:.6f}*V + {temp_coeffs[2]:.6f}*V^2")

