import numpy as np



T = np.array([66.25,64.40,63.50,62.20,61.20,60.00,58.80,57.54,55.77,54.45,51.85,50.20,49.00,45.06,44.34,43.9,42.8,40.8,39.80,38.76,37.23,36.58])
V = np.array([2.0283,2.0332,2.0381,2.0430,2.0479,2.0528,2.0577,2.0626,2.0674,2.0723,2.0821,2.0870,2.0919,2.114,2.1212,2.1310,2.1408,2.1652,2.1750,2.1799,2.1848,2.2092])

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
