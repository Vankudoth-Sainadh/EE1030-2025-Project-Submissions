import numpy as np
import matplotlib.pyplot as plt

# Your experimental data
temperatures = np.array([64.1, 66.8, 70.3, 74.9, 78.6, 82.4, 86.9, 89.6])
voltages = np.array([1.106, 1.119, 1.153, 1.153, 1.169, 1.185, 1.205, 1.217])

# Reshape for matrix operations
T = temperatures.reshape(-1, 1)
C = voltages.reshape(-1, 1)

# Create feature matrix for quadratic model: V = a + b*T + c*T^2
X = np.hstack((np.ones((T.shape[0], 1)), T, T**2))

# Least squares method to find coefficients
coefficients = np.linalg.lstsq(X, C, rcond=None)[0]
a, b, c = coefficients[0][0], coefficients[1][0], coefficients[2][0]

print("="*70)
print("QUADRATIC MODEL COEFFICIENTS (From Your Experimental Data)")
print("="*70)
print(f"Model: V = a + b*T + c*T²")
print(f"\na = {a:.10f}")
print(f"b = {b:.10f}")
print(f"c = {c:.10f}")
print("="*70)

# Scientific notation for better readability
print(f"\nScientific Notation:")
print(f"a = {a:.6e}")
print(f"b = {b:.6e}")
print(f"c = {c:.6e}")
print("="*70)

# Calculate R-squared value
predicted = X @ coefficients
residuals = C - predicted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((C - np.mean(C))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\nModel Quality Metrics:")
print(f"R² (coefficient of determination) = {r_squared:.10f}")
print(f"Root Mean Square Error (RMSE) = {np.sqrt(np.mean(residuals**2)):.6f} V")
print(f"Mean Absolute Error (MAE) = {np.mean(np.abs(residuals)):.6f} V")
print(f"Model fit quality: {'Excellent' if r_squared > 0.99 else 'Good' if r_squared > 0.95 else 'Fair'}")
print("="*70)

# Create a finer temperature range for smooth plotting
T_smooth = np.linspace(temperatures.min(), temperatures.max(), 100).reshape(-1, 1)
X_smooth = np.hstack((np.ones((T_smooth.shape[0], 1)), T_smooth, T_smooth**2))
predicted_smooth = X_smooth @ coefficients

# Plot the results
fig = plt.figure(figsize=(14, 5))

# Subplot 1: Training data and fitted curve
plt.subplot(1, 3, 1)
plt.plot(T_smooth, predicted_smooth, 'b-', linewidth=2.5, label='Fitted Quadratic Model')
plt.plot(T, C, 'ro', markersize=10, label='Experimental Data', markeredgecolor='darkred', markeredgewidth=1.5)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
plt.ylabel('Voltage (V)', fontsize=12, fontweight='bold')
plt.title('PT100 Training: Temperature vs Voltage', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.tight_layout()

# Subplot 2: Residuals (errors)
plt.subplot(1, 3, 2)
plt.plot(T, residuals * 1000, 'go-', linewidth=2, markersize=8, markeredgecolor='darkgreen', markeredgewidth=1.5)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
plt.ylabel('Residual Error (mV)', fontsize=12, fontweight='bold')
plt.title('Model Prediction Error', fontsize=13, fontweight='bold')

# Subplot 3: Data table visualization
plt.subplot(1, 3, 3)
plt.axis('off')
table_data = []
table_data.append(['Temp (°C)', 'Measured (V)', 'Predicted (V)', 'Error (mV)'])
for temp, volt, pred, res in zip(T, C, predicted, residuals):
    table_data.append([f'{temp[0]:.1f}', f'{volt[0]:.3f}', f'{pred[0]:.3f}', f'{res[0]*1000:.2f}'])

table = plt.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.22, 0.28, 0.28, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('Experimental Data vs Model', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('pt100_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Create training.csv file
print("\n" + "="*70)
print("Creating training.csv file...")
training_data = np.hstack((T, C))
np.savetxt('training.csv', training_data, delimiter=',', 
           fmt='%.6f', header='Temperature(C),Voltage(V)', comments='')
print("✓ training.csv created successfully!")
print("="*70)

# Display the detailed training data table
print("\nDetailed Training Data Analysis:")
print("-"*70)
print(f"{'Temp (°C)':<12} {'Measured (V)':<15} {'Predicted (V)':<15} {'Error (mV)':<12}")
print("-"*70)
for temp, volt, pred, res in zip(T, C, predicted, residuals):
    print(f"{temp[0]:<12.1f} {volt[0]:<15.6f} {pred[0]:<15.6f} {res[0]*1000:<12.3f}")
print("="*70)

# Temperature sensitivity
sensitivity = (voltages[-1] - voltages[0]) / (temperatures[-1] - temperatures[0])
print(f"\nSensor Sensitivity: {sensitivity*1000:.3f} mV/°C")
print(f"Temperature Range: {temperatures[0]:.1f}°C to {temperatures[-1]:.1f}°C")
print(f"Voltage Range: {voltages[0]:.3f}V to {voltages[-1]:.3f}V")
print("="*70)