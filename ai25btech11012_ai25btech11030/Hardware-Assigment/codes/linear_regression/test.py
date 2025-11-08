import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
training_data = pd.read_csv('training2.csv')
testing_data = pd.read_csv('testing.csv')

# Extract training data
v_train = training_data['V'].values
t_train = training_data['T'].values

# Fit quadratic curve using training data
coefficients = np.polyfit(v_train, t_train, 2)
a, b, c = coefficients

# Extract testing data
v_test = testing_data['V'].values
t_actual = testing_data['T'].values
t_predicted = testing_data['T_p'].values

# Generate smooth curve
v_curve = np.linspace(1.30, 1.50, 100)
t_curve = a * v_curve**2 + b * v_curve + c

# Create graph
plt.figure(figsize=(8, 6))
plt.plot(v_curve, t_curve, color='red', linewidth=2.5, label='Fitted Curve')
plt.scatter(v_test, t_actual, color='green', marker='s', s=100, label='Actual Test Points', zorder=5)
plt.scatter(v_test, t_predicted, color='orange', marker='^', s=100, label='Predicted Points', zorder=5)

plt.xlabel('V (Voltage)', fontsize=12)
plt.ylabel('T (Temperature)', fontsize=12)
plt.title('V vs T: Validation Curve vs Test Points', fontsize=13, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(1.30, 1.50)
plt.tight_layout()
plt.show()

