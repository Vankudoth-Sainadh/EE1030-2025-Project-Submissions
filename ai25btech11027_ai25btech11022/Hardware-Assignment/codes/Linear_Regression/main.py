import numpy as np
#import pandas as pd

#Loading Data from data.csv
# data = pd.read_csv("../data/train.csv")

T = np.array([88.5, 85.5, 81.2, 78.4, 74.0, 62.7, 66.6, 59.4, 55.3, 53.3, 46.0, 42.1, 40.5, 39.1, 27.6])
V = np.array([1.139, 1.134, 1.124, 1.116, 1.105, 1.075, 1.085, 1.070, 1.056, 1.035, 1.107, 1.124, 1.009, 0.992, 0.973])

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
