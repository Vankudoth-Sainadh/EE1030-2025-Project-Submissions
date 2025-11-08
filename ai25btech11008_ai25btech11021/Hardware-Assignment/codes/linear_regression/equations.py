#Code by Harshith Chiruvella and Abhiram Reddy

import numpy as np

#Load data from training.txt
data = np.loadtxt('training.txt)

T = A[:,[0]]
V = A[:,[1]]

#V = v + vb*T + va*T**2
X = np.vstack((np.ones_like(T), T, T**2)).T
coeffs = np.linalg.lstsq(X, V, rcond=None)[0]
 = coeffs

print("Voltage Model: V(T) = {:.6f} + {:.6e}*T + {:.6e}*T^2".format(v, vb, va1))

# T = m + l*V + k*V**2
X_inv = np.vstack((np.ones_like(V), V, V**2)).T
coeffs_inv = np.linalg.lstsq(X_inv, T, rcond=None)[0]
 = coeffs_inv

print("Temperature Model: T(V) = {:.6f} + {:.6f}*V + {:.6f}*V^2".format(m, l, k))