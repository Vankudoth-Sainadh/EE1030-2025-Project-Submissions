import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------
# Step 1. Coefficients from your training (update with yours)
# Equation: T = a0 + a1*V + a2*V^2
a0 = 1411.268101
a1 = -1708.333692
a2 = 530.375983

# ------------------------------------------------
# Step 2. Validation Data (from your table)
V_test = np.array([1.89, 1.88, 1.85, 1.84, 1.84, 1.83, 1.83, 1.82, 1.81, 1.79])
T_actual = np.array([72.90, 71.89, 65.43, 65.22, 64.22, 63.03, 61.88, 57.50, 55.00, 53.10])

# ------------------------------------------------
# Step 3. Predict temperature using the trained model
T_pred = a0 + a1*V_test + a2*(V_test**2)

# ------------------------------------------------
# Step 4. Create validation DataFrame
df = pd.DataFrame({
    "Voltage (V)": V_test,
    "Temperature Predicted (°C)": T_pred.round(2),
    "Temperature Actual (°C)": T_actual
})
print("\nValidation Table:")
print(df)

# ------------------------------------------------
# Step 5. Calculate model accuracy
mae = mean_absolute_error(T_actual, T_pred)
r2 = r2_score(T_actual, T_pred)
print(f"\nMean Absolute Error (MAE): {mae:.2f} °C")
print(f"R² Score: {r2:.3f}")

# ------------------------------------------------
# Step 6. Plot Validation Graph
plt.figure(figsize=(8,6))
plt.scatter(T_actual, V_test, color='red', label='Measured Data')
plt.plot(T_pred, V_test, color='black', label='Best Fit Curve')
plt.xlabel('Temperature (°C)')
plt.ylabel('Output Voltage (V)')
plt.title('Voltage vs Temperature (PT100 Sensor Calibration)')
plt.legend()
plt.grid(True)

plt.savefig("figs/Validation_Curve.png", dpi=300, bbox_inches='tight')
plt.show()
