

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data from CSV
df = pd.read_csv('../../tables/train.csv')
# Change the column names below as per your file if needed
T = df['T'].values       # Temperature column
V = df['V'].values       # Voltage column

# Fit a second-degree polynomial (quadratic)
coeffs = np.polyfit(T, V, 2)
a, b, c = coeffs
print(f"Best-fit curve: V = {a:.6e}*T^2 + {b:.6e}*T + {c:.6e}")

# Generate fitted values for a smooth curve
T_sorted = np.linspace(T.min(), T.max(), 200)
V_fit = np.polyval(coeffs, T_sorted)

# Set tick locations for clarity
xticks = np.arange(int(T.min()), int(T.max()) + 1, 5)

# Plot data and best-fit curve
plt.figure(figsize=(8, 5))
plt.scatter(T, V, color='r', zorder=5, edgecolor='k', marker='H', s=50, label='Measured data')
plt.plot(T_sorted, V_fit, color='b', label='Best-fit curve')

# Set axis labels and formatting
plt.xlabel('Temperature (°C)', fontsize=18)
plt.ylabel('Output Voltage (V)', fontsize=18)
plt.xticks(xticks, fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('../../figs/train.png', dpi=300)
plt.show()
