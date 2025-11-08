
import numpy as np
import pandas as pd


T = np.array([97.9, 97.8, 92.5, 91.6, 88.1, 86.1, 80.2, 73.9,
  71.3, 68.4, 66.3, 63.2, 60.3, 57.5, 55.0, 53.4,
  53.1, 52.1, 49.6, 46.8, 42.6, 40.3, 40.0, 39.1])
V = np.array([1.96, 1.95, 1.94, 1.93, 1.92, 1.91, 1.90, 1.89,
  1.87, 1.86, 1.85, 1.84, 1.83, 1.82, 1.81, 1.80,
  1.78, 1.77, 1.76, 1.75, 1.74, 1.72, 1.70, 1.69])

#V = n0 + n1*T + n2*T**2
X = np.vstack((np.ones_like(T), T, T**2)).T
coeffs = np.linalg.lstsq(X, V, rcond=None)[0]
n0, n1, n2 = coeffs

print("Voltage Model: V(T) = {:.6f} + {:.6e}*T + {:.6e}*T^2".format(n0, n1, n2))

# T = a0 + a1*V + a2*V**2
X_inv = np.vstack((np.ones_like(V), V, V**2)).T
coeffs_inv = np.linalg.lstsq(X_inv, T, rcond=None)[0]
a0, a1, a2 = coeffs_inv

print("Temperature Model: T(V) = {:.6f} + {:.6f}*V + {:.6f}*V^2".format(a0, a1, a2))

