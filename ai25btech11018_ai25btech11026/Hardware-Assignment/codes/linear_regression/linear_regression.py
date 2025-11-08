import numpy as np
import pandas as pd

V = np.array([2.121, 2.126, 2.131, 2.136, 2.14, 2.146, 2.151, 2.155, 2.160, 2.165, 2.170, 2.185, 2.195, 2.199, 2.204, 2.209, 2.214, 2.219, 2.322, 2.366, 2.375, 2.507, 2.62, 2.722, 2.727, 2.742
])
T = np.array([85, 81.6, 80, 79, 77.7, 76.5, 75.2, 73.5, 71.4, 70.4, 69.6, 66.6, 65.5, 64.5, 63, 61.4, 59.3, 58.5, 53, 52.5, 51.9, 50.5, 46.7, 42.7, 36.4, 29
])

#V = n0 + n1*T + n2*T**2
X = np.vstack((np.ones_like(T), T, T**2)).T
coeffs = np.linalg.lstsq(X, V, rcond=None)[0]
n0, n1, n2 = coeffs
print("Voltage Model: V(T) = {:.6f} + {:.6f}*T + {:.6e}*T^2".format(n0, n1, n2))

#T = a0 + a1*V + a2*V**2
X_inv = np.vstack((np.ones_like(V), V, V**2)).T
coeffs_inv = np.linalg.lstsq(X_inv, T, rcond=None)[0]
a0, a1, a2 = coeffs_inv
print("Temperature Model: T(V) = {:.6f} + {:.6f}*V + {:.6f}*V^2".format(a0, a1, a2))
