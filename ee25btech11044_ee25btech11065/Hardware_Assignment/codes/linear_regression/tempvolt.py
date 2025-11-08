import numpy as np
import matplotlib.pyplot as plt

# Your experimental data
temperatures = np.array([64.1, 66.8, 70.3, 74.9, 78.6, 82.4, 86.9, 89.6])
voltages = np.array([1.106, 1.119, 1.153, 1.153, 1.169, 1.185, 1.205, 1.217])

print("="*70)
print("INVERSE MODEL: VOLTAGE → TEMPERATURE")
print("="*70)

# ==================== FORWARD MODEL ====================
# First, get the forward model coefficients (V = a + b*T + c*T²)
T = temperatures.reshape(-1, 1)
V = voltages.reshape(-1, 1)

X_forward = np.hstack((np.ones((T.shape[0], 1)), T, T**2))
coef_forward = np.linalg.lstsq(X_forward, V, rcond=None)[0]
a, b, c = coef_forward[0][0], coef_forward[1][0], coef_forward[2][0]

print("\nForward Model Coefficients:")
print(f"V = a + b*T + c*T²")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.10f}")

# ==================== INVERSE MODEL ====================
# Now create inverse model (T = p + q*V + r*V²)
X_inverse = np.hstack((np.ones((V.shape[0], 1)), V, V**2))
coef_inverse = np.linalg.lstsq(X_inverse, T, rcond=None)[0]
p, q, r = coef_inverse[0][0], coef_inverse[1][0], coef_inverse[2][0]

print("\n" + "="*70)
print("INVERSE MODEL COEFFICIENTS:")
print("="*70)
print(f"Model: T = p + q*V + r*V²")
print(f"\np = {p:.3f}")
print(f"q = {q:.3f}")
print(f"r = {r:.3f}")
print("\n" + "="*70)

print(f"\nScientific Notation:")
print(f"p = {p:.6e}")
print(f"q = {q:.6e}")
print(f"r = {r:.6e}")
print("="*70)

# Calculate R-squared for inverse model
predicted_temp = X_inverse @ coef_inverse
residuals_inv = T - predicted_temp
ss_res_inv = np.sum(residuals_inv**2)
ss_tot_inv = np.sum((T - np.mean(T))**2)
r_squared_inv = 1 - (ss_res_inv / ss_tot_inv)

print(f"\nInverse Model Quality:")
print(f"R² = {r_squared_inv:.10f}")
print(f"RMSE = {np.sqrt(np.mean(residuals_inv**2)):.3f} °C")
print(f"MAE = {np.mean(np.abs(residuals_inv)):.3f} °C")
print("="*70)

# ==================== ANALYTICAL SOLUTION (for linear approximation) ====================
if abs(c) < 1e-6 and abs(r) < 1:  # If quadratic terms are negligible
    print("\n" + "="*70)
    print("LINEAR APPROXIMATION (Simplified):")
    print("="*70)
    print(f"Since c ≈ 0, we can use linear approximation:")
    print(f"V ≈ {a:.6f} + {b:.6f}*T")
    print(f"\nInverse (solving for T):")
    print(f"T ≈ (V - {a:.6f}) / {b:.6f}")
    T_linear_coef = 1/b
    T_linear_intercept = -a/b
    print(f"T ≈ {T_linear_intercept:.3f} + {T_linear_coef:.3f}*V")
    print("="*70)

# ==================== VISUALIZATION ====================
fig = plt.figure(figsize=(15, 5))

# Subplot 1: Inverse relationship (V to T)
plt.subplot(1, 3, 1)
V_smooth = np.linspace(voltages.min(), voltages.max(), 100).reshape(-1, 1)
X_inv_smooth = np.hstack((np.ones((V_smooth.shape[0], 1)), V_smooth, V_smooth**2))
T_predicted_smooth = X_inv_smooth @ coef_inverse

plt.plot(V_smooth, T_predicted_smooth, 'r-', linewidth=2.5, label='Inverse Model')
plt.plot(V, T, 'bo', markersize=10, label='Experimental Data', markeredgecolor='darkblue', markeredgewidth=1.5)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Voltage (V)', fontsize=12, fontweight='bold')
plt.ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
plt.title('Inverse Model: Voltage → Temperature', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)

# Subplot 2: Residual errors
plt.subplot(1, 3, 2)
plt.plot(V, residuals_inv, 'mo-', linewidth=2, markersize=8, markeredgecolor='darkmagenta', markeredgewidth=1.5)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel('Voltage (V)', fontsize=12, fontweight='bold')
plt.ylabel('Temperature Error (°C)', fontsize=12, fontweight='bold')
plt.title('Inverse Model Prediction Error', fontsize=13, fontweight='bold')

# Subplot 3: Comparison table
plt.subplot(1, 3, 3)
plt.axis('off')
table_data = []
table_data.append(['Voltage (V)', 'Actual T (°C)', 'Predicted T (°C)', 'Error (°C)'])
for volt, temp, pred, res in zip(V, T, predicted_temp, residuals_inv):
    table_data.append([f'{volt[0]:.3f}', f'{temp[0]:.1f}', f'{pred[0]:.1f}', f'{res[0]:.2f}'])

table = plt.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.28, 0.28, 0.19])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#C00000')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F2F2F2')

plt.title('Inverse Model Validation', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('pt100_inverse_model.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== TEST THE INVERSE MODEL ====================
print("\n" + "="*70)
print("TESTING INVERSE MODEL:")
print("="*70)
print(f"{'Input Voltage (V)':<20} {'Predicted Temperature (°C)':<30}")
print("-"*70)

# Test with the experimental voltages
for volt in voltages:
    T_pred = p + q*volt + r*volt**2
    print(f"{volt:<20.3f} {T_pred:<30.2f}")

print("="*70)

# ==================== PRACTICAL USAGE FUNCTION ====================
print("\n" + "="*70)
print("PYTHON FUNCTION FOR PRACTICAL USE:")
print("="*70)
print(f"""
def voltage_to_temperature(voltage):
    '''
    Convert PT100 voltage reading to temperature
    
    Args:
        voltage: Measured voltage in Volts
    
    Returns:
        temperature: Temperature in Celsius
    '''
    p = {p:.6f}
    q = {q:.6f}
    r = {r:.6f}
    
    temperature = p + q*voltage + r*voltage**2
    return temperature

# Example usage:
measured_voltage = 1.150  # V
temperature = voltage_to_temperature(measured_voltage)
print(f"Temperature: {{temperature:.2f}}°C")
""")
print("="*70)