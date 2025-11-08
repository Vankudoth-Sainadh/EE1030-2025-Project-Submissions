import numpy as np

a0 = 2084.186654
a1 = -1216.292546
a2 = 175.759890

data = np.loadtxt("data2.csv", delimiter=',', skiprows=1)
T_measured = data[:, 0]
V = data[:, 1]

T_pred = a0 + a1 * V + a2 * (V ** 2)

error = T_measured - T_pred
mean_error = np.mean(np.abs(error))  

print("=== Mean Error Analysis ===")
print(f"a0 = {a0:.6f}, a1 = {a1:.6f}, a2 = {a2:.6f}")
print(f"Mean Absolute Error = {mean_error:.6f} °C")

# Optional: also show max/min error
print(f"Max Error = {np.max(error):.6f} °C")
print(f"Min Error = {np.min(error):.6f} °C")

