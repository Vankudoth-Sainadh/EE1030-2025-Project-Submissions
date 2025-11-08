import numpy as np

# Load data from CSV file
# File should have two columns: T and V (with or without header)
data = np.loadtxt("/home/dhanush-kumar-a/EE1030-2025-Project-Submissions/ai25btech11010_ai25btech11014/Hardware-Assignment/tables/data1.csv", delimiter=',', skiprows=1)  # skip header line if present

T = data[:, 0]
V = data[:, 1]

# Model: V = n0 + n1*T + n2*T^2
X = np.vstack((np.ones_like(T), T, T**2)).T
n0, n1, n2 = np.linalg.lstsq(X, V, rcond=None)[0]

print(f"Voltage Model: V(T) = {n0:.6f} + {n1:.6e}*T + {n2:.6e}*T^2")

# Inverse model: T = a0 + a1*V + a2*V^2
X_inv = np.vstack((np.ones_like(V), V, V**2)).T
a0, a1, a2 = np.linalg.lstsq(X_inv, T, rcond=None)[0]

print(f"Temperature Model: T(V) = {a0:.6f} + {a1:.6f}*V + {a2:.6f}*V^2")

