import numpy as np
import pandas as pd

# Loading Data from train.csv
data = pd.read_csv("../data/train.csv")

T = np.array(data["T"])
V = np.array(data["V"])

# Fit 4th-degree polynomial: V = n0 + n1*T + n2*T^2 + n3*T^3 + n4*T^4
X = np.vstack((np.ones_like(T), T, T**2, T**3, T**4)).T
coeffs = np.linalg.lstsq(X, V, rcond=None)[0]
n0, n1, n2, n3, n4 = coeffs

print("Voltage Model: V(T) = {:.6f} + {:.6e}*T + {:.6e}*T^2 + {:.6e}*T^3 + {:.6e}*T^4"
      .format(n0, n1, n2, n3, n4))

# Fit inverse 4th-degree polynomial: T = a0 + a1*V + a2*V^2 + a3*V^3 + a4*V^4
X_inv = np.vstack((np.ones_like(V), V, V**2, V**3, V**4)).T
coeffs_inv = np.linalg.lstsq(X_inv, T, rcond=None)[0]
a0, a1, a2, a3, a4 = coeffs_inv

print("Temperature Model: T(V) = {:.6f} + {:.6e}*V + {:.6e}*V^2 + {:.6e}*V^3 + {:.6e}*V^4"
      .format(a0, a1, a2, a3, a4))