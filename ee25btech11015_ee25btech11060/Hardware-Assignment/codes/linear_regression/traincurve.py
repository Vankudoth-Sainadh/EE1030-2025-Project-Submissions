import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading your data
df = pd.read_csv("../data/train.csv")

T = df["T"].values
V = df["V"].values

# Best Fit Curve (biquadratic: 4th degree)
# V = a*T^4 + b*T^3 + c*T^2 + d*T + e
coeffs = np.polyfit(T, V, 4)
a, b, c, d, e = coeffs
print(f"Best-fit curve: V = {a:.6e}·T⁴ + {b:.6e}·T³ + {c:.6e}·T² + {d:.6e}·T + {e:.6e}")

# Computing fitted values for plotting
T_sorted = np.linspace(T.min(), T.max(), 200)
V_fit = np.polyval(coeffs, T_sorted)

# Ticks
x_ticks = np.arange(25, 82, 2)

# Plotting the raw data and the best-fit curve
plt.figure(figsize=(8,5))
plt.scatter(T, V, color="r", zorder=5, edgecolor="k", marker="H", s=50, label="Measured data")
plt.plot(T_sorted, V_fit, color="b", label="Biquadratic best-fit curve")

# Labelling and adjusting axes
plt.xlabel("Temperature (°C)", fontsize=25)
plt.ylabel("Output Voltage (V)", fontsize=25)
plt.xticks(x_ticks)

# Legend, grid, and layout
plt.legend()
plt.grid(True)
plt.tight_layout()

# Saving and showing the image
plt.savefig("../figs/fig1_biquadratic.png", dpi=300)
plt.show()
