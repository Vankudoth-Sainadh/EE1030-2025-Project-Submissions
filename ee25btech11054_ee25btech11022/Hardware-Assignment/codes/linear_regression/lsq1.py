import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('../../tables/training_data1.txt')
X = np.hstack((np.ones((A.shape[0], 1)), A[:, [0]], A[:, [0]]**2))
T = A[:, [1]]   # Temperature (°C)
C = A[:, [0]]   # Voltage (V)

# Least squares method
c0, c1, c2 = np.linalg.lstsq(X, T, rcond=None)[0]
n_lsq = np.zeros((3, 1))
n_lsq[0][0] = c0
n_lsq[1][0] = c1
n_lsq[2][0] = c2
print(n_lsq)

# Plot training data fit
plt.plot(C, X @ n_lsq, label='Fitted curve')
plt.plot(C, T, 'k.', label='Training data')
plt.xlabel('Output Voltage (V)')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('../../figs/train1.png')
plt.close('all')

# Validation data
B = np.loadtxt('../../tables/validation_data1.txt')
Xv = np.hstack((np.ones((B.shape[0], 1)), B[:, [0]], B[:, [0]]**2))
Cv = B[:, [0]]
Tv = B[:, [1]]

plt.plot(Cv, Xv @ n_lsq, label='Predicted')
plt.plot(Cv, Tv, 'k.', label='Validation data')
plt.xlabel('Output Voltage (V)')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('../../figs/valid1.png')
plt.close('all')

