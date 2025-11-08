import numpy as np
# import pandas as pd

#Loading Data from data.csv
# data = pd.read_csv("../data/train.csv")

T = np.array([26.5, 33.6, 37.5, 41.2, 47.6, 53.5, 57.2, 62.5, 67.4, 72.1, 77.6, 82.5, 89.5, 94.8, 30.5, 36.4, 40.2, 43.4])
V = np.array([2.224, 2.204, 2.19, 2.18, 2.155, 2.126, 2.092, 2.067, 2.058, 2.038, 2.023, 1.994, 1.979, 1.955, 2.214, 2.224, 2.185, 2.18])

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
