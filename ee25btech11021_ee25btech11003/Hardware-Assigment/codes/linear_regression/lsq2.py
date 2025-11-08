

import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training_data.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[1]],A[:,[1]]**2))
T = A[:,[0]]
V = A[:,[1]]

#Least squares method
v, av, bv = np.linalg.lstsq(X, T, rcond=None)[0]
n_lsq = np.zeros((3,1))
n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv
print(n_lsq)

#Plot both the results
plt.plot(V, X@n_lsq)
plt.plot(V, T, 'k.')
plt.ylabel('Temperature ($^{\circ}$C)')
plt.xlabel('Output Voltage (V)')
plt.grid()
plt.tight_layout()
plt.savefig('../figs/train1.png')

#Close current figure(s)
plt.close('all')

#Plot for validation
B = np.loadtxt('validation_data.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[1]],B[:,[1]]**2))
Vv = B[:,[1]]
Tv = B[:,[0]]
plt.plot(Vv, Xv@n_lsq)
plt.plot(Vv, Tv, 'k.')
plt.ylabel('Temperature ($^{\circ}$C)')
plt.xlabel('Output Voltage (V)')
plt.grid()
plt.tight_layout()
plt.savefig('../figs/valid1.png')

