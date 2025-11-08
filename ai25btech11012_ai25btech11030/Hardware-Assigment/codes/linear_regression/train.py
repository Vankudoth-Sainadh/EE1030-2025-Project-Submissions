import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load training data
training_data = pd.read_csv('training2.csv')

# Extract training data
v_train = training_data['V'].values
t_train = training_data['T'].values

# Fit quadratic curve (T as function of V)
coefficients = np.polyfit(v_train, t_train, 2)
a, b, c = coefficients

# Generate smooth curve
v_curve = np.linspace(1.30, 1.50, 100)
t_curve = a * v_curve**2 + b * v_curve + c

# Create graph
plt.figure(figsize=(8, 6))
plt.scatter(v_train, t_train, color='blue', s=80, label='Data Points', zorder=5)
plt.plot(v_curve, t_curve, color='red', linewidth=2.5, label='Fitted Curve')

plt.xlabel('V (Voltage)', fontsize=12)
plt.ylabel('T (Temperature)', fontsize=12)
plt.title('V vs T: Quadratic Curve Fit', fontsize=13, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

