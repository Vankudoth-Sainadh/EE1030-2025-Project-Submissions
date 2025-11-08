#Code by Harshith Chiruvella and Abhiram Reddy

import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2))
T = A[:,[0]]
V = A[:,[1]]

#Least squares method
v, av, bv = np.linalg.lstsq(X, V, rcond=None)[0]
n_lsq = np.zeros((3,1))
n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv
print("LSQ", n_lsq)


coeff = np.linalg.inv(X.T @ X) @ X.T @ V
print("Coeff", coeff)

print(n_lsq - coeff)

#Plot both the results
plt.plot(T, X@n_lsq)
plt.plot(T, V, 'k.')
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\\circ}$C)')
plt.tight_layout()
plt.savefig('../figs/model.png')